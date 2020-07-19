from abc import ABC, abstractmethod
import Logger
import zmq
import zmq.asyncio
import pickle
import asyncio
from sanic import Sanic
from sanic import response
import pandas as pd
import time
import os
from FutuHook import d_types
import datetime
import requests
import itertools
import shutil
import collections
import json


class BaseAlgo(ABC):
    def __init__(self, name: str, benchmark: str = 'SPX'):
        # settings
        self.name = name
        self.benchmark = benchmark

        self.logger = Logger.RootLogger(root_name=self.name)

        self._trading_environment = ''
        self._trading_universe = None
        self._failed_tickers = None
        self._datatypes = None
        self._txn_cost = None
        self._total_txn_cost = 0
        self._initial_capital = 0.0
        self._margin = 0.0

        # current status
        self._running = False
        self._current_cash = 0.0
        self._current_margin = 0.0

        # Info
        self.pending_orders = dict()
        self.completed_orders = pd.DataFrame(columns=['order_id'])
        self.positions = pd.DataFrame(columns=['price', 'quantity', 'market_value'])
        self.slippage = pd.DataFrame(columns=['exp_price', 'dealt_price', 'dealt_qty', 'total_slippage'])
        self.ticker_lot_size = None

        self.record = pd.DataFrame(columns=['PV', 'EV', 'Cash', 'Margin'])

        # IPs
        self._ip = ''
        self._mq_ip = ''
        self._hook_ip = ''
        self._zmq_context = None
        self._mq_socket = None
        self._topics = None
        self._hook_name = None

        # Cache
        self.cache_path = None
        self.cache = None
        self._max_cache = 0
        self._ticker_max_cache = 0
        self._cache_retain_ratio = 0
        self.cache_pickle_no_map = collections.defaultdict(lambda: 1)

        self.initialized_date = datetime.datetime.today()

        # Web
        self._sanic = None
        self._sanic_host = None
        self._sanic_port = None
        self._initialized = False

    def initialize(self, initial_capital: float, margin: float, mq_ip: str,
                   hook_ip: str, hook_name: str, trading_environment: str,
                   trading_universe: list, datatypes: list,
                   txn_cost: float = 30, max_cache: int = 50000, ticker_max_cache: int = 3000,
                   cache_retain_ratio: float = 0.3, test_mq_con=True, **kwargs):
        try:
            assert trading_environment in ('REAL', 'SIMULATE', 'BACKTEST'), 'Invalid trading universe {}'.format(
                trading_environment)

            for dtype in datatypes:
                assert dtype in d_types, 'Invalid data type {}'.format(dtype)

            assert margin >= initial_capital, 'Margin must be >= intial capital'

            self._trading_environment = trading_environment
            self._trading_universe = trading_universe
            self._datatypes = datatypes
            self._txn_cost = txn_cost
            self._initial_capital = initial_capital
            self._margin = margin
            self._mq_ip = mq_ip
            self._hook_ip = hook_ip
            self._hook_name = hook_name
            self._current_margin = 0.0
            self._current_cash = initial_capital

            if test_mq_con:
                # Test Connection with ZMQ
                test_context = zmq.Context()
                try:
                    test_socket = test_context.socket(zmq.PAIR)
                    test_socket.setsockopt(zmq.LINGER, 0)
                    test_socket.setsockopt(zmq.SNDTIMEO, 2000)
                    test_socket.setsockopt(zmq.RCVTIMEO, 2000)

                    hello_mq_ip = self._mq_ip.split(':')
                    hello_mq_ip = ':'.join([hello_mq_ip[0], hello_mq_ip[1], str(int(hello_mq_ip[2]) + 1)])
                    test_socket.connect(hello_mq_ip)
                    test_socket.send_string('Ping')
                    msg = test_socket.recv_string()
                    if msg != 'Pong':
                        raise Exception('no Pong!')
                    self.logger.info(f'Test Connection with ZMQ {self._mq_ip} is Successful!')
                    self.logger.info(f'Test Connection with ZMQ {hello_mq_ip} is Successful!')

                except zmq.error.Again:
                    raise Exception(f'Failed to connect to ZMQ, please check : {self._mq_ip}')
                # except Exception as e:
                #     raise Exception(f'Failed to connect to ZMQ, please check : {self._mq_ip}, reason: {str(e)}')

                finally:
                    test_context.destroy()

            # Test Connection with Hook
            try:
                requests.get(self._hook_ip + '/subscriptions').json()
                self.logger.info(f'Test Connection with Hook IP f{self._hook_ip} is Successful!')
            except requests.ConnectionError:
                raise Exception(f'Failed to connect to Hook, please check: {self._hook_ip}')

            # Subscription data
            self.ticker_lot_size = dict()
            self._failed_tickers = list()
            self._topics = list()

            # Cache
            self.cache_path = './{}_cache'.format(self.name)
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
            os.mkdir(self.cache_path)
            self.cache = dict()
            for datatype in d_types:
                self.cache[datatype] = pd.DataFrame()
            self._max_cache = max_cache
            self._ticker_max_cache = ticker_max_cache
            self._cache_retain_ratio = cache_retain_ratio
            self.cache_pickle_no_map = collections.defaultdict(lambda: 1)

            self._initialized = True
            self.logger.info('Initialized sucessfully.')

        except Exception as e:
            self._trading_environment = ''
            self._trading_universe = None
            self._datatypes = None
            self._txn_cost = None
            self._initial_capital = 0.0
            self._margin = 0.0
            self._total_txn_cost = 0

            self._running = False
            self._current_cash = 0.0
            self._current_margin = 0.0

            self._mq_ip = ''
            self._hook_ip = ''
            self._hook_name = None

            self._failed_tickers = list()
            self.ticker_lot_size = dict()
            self._topics = list()

            self.logger.error(f'Failed to initialize algo, reason: {str(e)}')
            self._initialized = False

    async def daily_record_performance(self):
        while True:
            self.log()
            await asyncio.sleep(60 * 60 * 24 - time.time() % 60 * 60 * 24)

    def log(self, date=None):
        EV = sum(self.positions['market_value'])
        self._current_margin = sum(abs(self.positions['market_value']))
        PV = EV + self._current_cash
        d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if date is None else date
        self.record.loc[d] = [PV, EV, self._current_cash, self._current_margin]

    async def main(self):

        self._zmq_context = zmq.asyncio.Context()
        self._mq_socket = self._zmq_context.socket(zmq.SUB)
        self._mq_socket.connect(self._mq_ip)
        self.subscribe_tickers(tickers=self._trading_universe)

        # for ticker in self._trading_universe:
        #     self.positions.loc[ticker] = [0.0, 0.0, 0.0]

        self._running = True
        self.logger.info(f'Algo {self.name} running successfully!')

        while True:

            try:
                topic, bin_df = await self._mq_socket.recv_multipart()
                if not self._running:
                    await asyncio.sleep(0.5)
                    continue

                topic_split = topic.decode('ascii').split('.')
                datatype = topic_split[1]
                key = '.'.join(topic_split[2:])
                df = pickle.loads(bin_df)
                if datatype == 'ORDER_UPDATE':
                    self.update_positions(df)
                    await self.on_order_update(order_id=key, df=df)
                else:
                    self.update_prices(datatype=datatype, df=df)
                    # TODO: improve add_cache places, determine_trigger, trigger_strat
                    self.add_cache(datatype=datatype, df=df)
                    trigger_strat, (tgr_dtype, tgr_ticker, tgr_df) = self.determine_trigger(datatype=datatype,
                                                                                            ticker=key, df=df)
                    if trigger_strat:
                        await self.trigger_strat(datatype=tgr_dtype, ticker=tgr_ticker, df=tgr_df)
            except Exception:
                self._running = False
                raise

    def determine_trigger(self, datatype, ticker, df):
        # TODO: improve add_cache places, determine_trigger, trigger_strat
        return True, (datatype, ticker, df)

    async def trigger_strat(self, datatype, ticker, df):
        # TODO: improve add_cache places, determine_trigger, trigger_strat
        if datatype == 'TICKER':
            await self.on_tick(ticker=ticker, df=df)
        elif datatype == 'QUOTE':
            await self.on_quote(ticker=ticker, df=df)
        elif 'K_' in datatype:
            await self.on_bar(datatype=datatype, ticker=ticker, df=df)
        elif datatype == 'ORDER_BOOK':
            await self.on_orderbook(ticker=ticker, df=df)
        else:
            await self.on_other_data(datatype=datatype, ticker=ticker, df=df)

    @abstractmethod
    async def on_other_data(self, datatype, ticker, df):
        pass

    @abstractmethod
    async def on_tick(self, ticker, df):
        pass

    @abstractmethod
    async def on_quote(self, ticker, df):
        pass

    @abstractmethod
    async def on_orderbook(self, ticker, df):
        pass

    @abstractmethod
    async def on_bar(self, datatype, ticker, df):
        pass

    @abstractmethod
    async def on_order_update(self, order_id, df):
        pass

    def run(self, sanic_port, sanic_host='127.0.0.1'):

        if not self._initialized:
            self.logger.info('Algo not initialized')
        else:
            loop = asyncio.get_event_loop()
            self._sanic = Sanic(self.name)
            self._sanic_host = sanic_host
            self._sanic_port = sanic_port
            self._ip = 'http://' + self._sanic_host + ':' + str(self._sanic_port)

            async def _run():

                tasks = list()
                self.app_add_route(app=self._sanic)

                web_server = self._sanic.create_server(return_asyncio_server=True, host=sanic_host, port=sanic_port)
                tasks.append(web_server)
                tasks.append(self.main())
                tasks.append(self.daily_record_performance())
                await asyncio.gather(*tasks)

            loop.create_task(_run())
            loop.run_forever()

    # ------------------------------------------------ [ Position ] ------------------------------------------
    def update_positions(self, df):

        # record order df
        trd_side = 1 if df['trd_side'].iloc[0].upper() in ('BUY', 'BUY_BACK') else -1
        dealt_qty = df['dealt_qty'].iloc[0] * trd_side
        avg_price = df['dealt_avg_price'].iloc[0]
        order_id = df['order_id'].iloc[0]
        ticker = df['ticker'].iloc[0]
        order_status = df['order_status'].iloc[0]

        in_pending = False
        if order_id in self.pending_orders.keys():
            in_pending = True
            last_order_update_df = self.pending_orders[order_id]

            last_qty = last_order_update_df['dealt_qty'].iloc[0] * trd_side
            last_avg_price = last_order_update_df['dealt_avg_price'].iloc[0]

            cash_change = -(dealt_qty * avg_price - last_qty * last_avg_price)
            qty_change = dealt_qty - last_qty

        else:
            if order_id not in self.completed_orders['order_id'].values:
                cash_change = - dealt_qty * avg_price
                qty_change = dealt_qty
            else:
                cash_change = 0
                qty_change = 0

        if order_status in ('SUBMIT_FAILED', 'FILLED_ALL', 'CANCELLED_PART', 'CANCELLED_ALL', 'FAILED', 'DELETED'):
            self.completed_orders = self.completed_orders.append(df).drop_duplicates('order_id', keep='last')

            # update slippage
            self.slippage.loc[order_id] = [0, 0, 0, 0]
            exp_price = self.positions.loc[ticker]['price'] if df['price'].iloc[0] == 0.0 else df['price'].iloc[0]
            self.slippage.loc[order_id] = [exp_price, avg_price, dealt_qty, (avg_price - exp_price) * dealt_qty]
            if in_pending:
                self._total_txn_cost += self._txn_cost
                if order_status in ('FILLED_ALL', 'CANCELLED_PART'):
                    cash_change -= self._txn_cost
                del self.pending_orders[order_id]

        else:

            self.pending_orders[order_id] = df

        # update positions and snapshot
        latest_price = self.positions.loc[ticker]['price']
        existing_qty = self.positions.loc[ticker]['quantity']

        new_qty = existing_qty + qty_change
        self.positions.loc[ticker] = [latest_price, new_qty, new_qty * latest_price]
        self._current_cash += cash_change

    def update_prices(self, datatype, df):
        if 'K_' in datatype:
            ticker = df['ticker'].iloc[0]
            qty = self.positions.loc[ticker]['quantity']
            latest_price = df['close'].iloc[0]
            self.positions.loc[ticker] = [latest_price, qty, qty * latest_price]

        elif datatype == 'QUOTE':
            ticker = df['ticker'].iloc[0]
            qty = self.positions.loc[ticker]['quantity']
            latest_price = df['quote'].iloc[0]
            self.positions.loc[ticker] = [latest_price, qty, qty * latest_price]

    # ------------------------------------------------ [ Data ] ------------------------------------------
    def add_cache(self, datatype, df):
        self.cache[datatype] = self.cache[datatype].append(df).drop_duplicates(
            subset=['datetime', 'ticker'], keep='last')

        if self.cache[datatype].shape[0] >= self._max_cache:
            drop_rows_n = int(self._max_cache * self._cache_retain_ratio)
            # Drop cache to pickle
            self.cache[datatype] = self.cache[datatype].sort_values('datetime')
            df_to_drop = self.cache[datatype].iloc[:drop_rows_n]
            self.cache[datatype] = self.cache[datatype][drop_rows_n:]
            for ticker in df_to_drop['ticker'].unique():
                ticker_df = df_to_drop.loc[df_to_drop['ticker'] == ticker]

                key = f'{datatype}_{ticker}'
                while ticker_df.shape[0] != 0:
                    try:
                        with open(f'{self.cache_path}/{key}_{self.cache_pickle_no_map[key]}.pickle',
                                  'rb+') as file:
                            db_df = pickle.load(file)
                            to_store = ticker_df.iloc[
                                       :min(drop_rows_n, self._ticker_max_cache - db_df.shape[0])]
                            ticker_df = ticker_df.iloc[to_store.shape[0]:]
                            to_store = db_df.append(to_store)
                            # to_store = pd.concat([db_df, to_store], axis=0)
                            file.truncate(0)
                            file.seek(0)
                            pickle.dump(to_store, file)
                    except (FileExistsError, FileNotFoundError):
                        with open(f'{self.cache_path}/{key}_{self.cache_pickle_no_map[key]}.pickle',
                                  'wb') as file:
                            to_store = ticker_df.iloc[:min(self._ticker_max_cache, ticker_df.shape[0])]
                            ticker_df = ticker_df.iloc[to_store.shape[0]:]
                            pickle.dump(to_store, file)

                    if to_store.shape[0] == self._ticker_max_cache:
                        self.cache_pickle_no_map[key] += 1
                        with open(f'{self.cache_path}/{key}_{self.cache_pickle_no_map[key]}.pickle',
                                  'wb') as file:
                            pickle.dump(pd.DataFrame(), file)

    def get_data(self, datatype, ticker: str, start_date: datetime.datetime):
        try:
            df = self.cache[datatype].loc[self.cache[datatype]['ticker'] == ticker]
            if (df.shape[0] > 0) and (min(df['datetime']) <= start_date):
                df = df.loc[df['datetime'] >= start_date]
                return df
        except KeyError:
            df = pd.DataFrame()

        key = f'{datatype}_{ticker}'
        last_id = self.cache_pickle_no_map[key]
        while last_id >= 1:
            try:
                with open(f'{self.cache_path}/{key}_{last_id}.pickle', 'rb') as file:
                    last_id -= 1
                    db_df = pickle.load(file)
                    df = db_df.append(df)
                    # df = pd.concat([db_df, df], axis=0)
                    if min(df['datetime']) <= start_date:
                        df = df.loc[df['datetime'] >= start_date]
                        break
            except (FileExistsError, FileNotFoundError, EOFError):
                last_id -= 1
                continue

        return df

    def get_data_rows(self, datatype, ticker: str, n_rows: int):
        try:
            df = self.cache[datatype].loc[self.cache[datatype]['ticker'] == ticker]
            if df.shape[0] >= n_rows:
                df = df.iloc[-n_rows:]
                return df
        except KeyError:
            df = pd.DataFrame()

        key = f'{datatype}_{ticker}'
        last_id = self.cache_pickle_no_map[key]
        while last_id >= 1:
            try:
                with open(f'{self.cache_path}/{key}_{last_id}.pickle', 'rb') as file:
                    last_id -= 1
                    db_df = pickle.load(file)
                    df = db_df.append(df)
                    # df = pd.concat([db_df, df], axis=0)
                    if df.shape[0] >= n_rows:
                        df = df.sort_values('datetime').iloc[-n_rows:]
                        break
            except (FileExistsError, FileNotFoundError, EOFError):
                last_id -= 1
                continue

        return df

    def download_historical(self, ticker, datatype, start_date, end_date, from_exchange=False):
        params = {'ticker': ticker, 'datatype': datatype, 'start_date': start_date, 'end_date': end_date,
                  'from_exchange': from_exchange}
        result = requests.get(self._hook_ip + '/historicals', params=params).json()
        if result['ret_code'] == 1:
            df = pd.read_json(result['return']['content'])
            return 1, df
        else:
            return 0, pd.DataFrame()

    def load_ticker_cache(self, ticker, datatype,
                          start_date=(datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
                          from_exchange=False):
        end_date = datetime.datetime.today().strftime('%Y-%m-%d')
        ret_code, df = self.download_historical(ticker=ticker, datatype=datatype, start_date=start_date,
                                                end_date=end_date, from_exchange=from_exchange)

        if ret_code == 1:
            self.add_cache(datatype=datatype, df=df)
        else:
            raise Exception('Failed to download historical data from Hook')

    def load_all_cache(self, tickers=None,
                       start_date=(datetime.datetime.today() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')):

        tickers = self._trading_universe if tickers is None else tickers
        for dtype in self._datatypes:
            for ticker in tickers:
                self.load_ticker_cache(ticker=ticker, datatype=dtype, start_date=start_date)

    def load_ticker_lot_size(self, tickers):
        params = {'tickers': str(tickers)}

        result = requests.get(self._hook_ip + '/order/lot_size', params=params)
        if result.status_code == 500:
            raise Exception(f'Format of one/more tickers are wrong, please check: {tickers}')
        result = result.json()
        if result['ret_code'] == 1:
            lot_size_df = pd.read_json(result['return']['content'])
            failed = list(lot_size_df.loc[lot_size_df['lot_size'] == 0].index)
            succeed = lot_size_df.loc[lot_size_df['lot_size'] > 0].to_dict()['lot_size']
            for ticker in succeed.keys():
                self.ticker_lot_size[ticker] = succeed[ticker]
            self._trading_universe = list(set(self._trading_universe).difference(failed))
            self._failed_tickers = self._failed_tickers + failed
            return list(succeed.keys()), failed
        else:
            raise Exception(f'Failed to request lot size, reason: {result["return"]["content"]}')

    def add_new_topics(self, tickers):
        tmp = [f'{self._hook_name}.{x[0]}.{x[1]}' for x in itertools.product(self._datatypes, tickers)]
        self._topics = list(set(self._topics).union(tmp))
        return tmp

    # ------------------------------------------------ [ Trade ] ------------------------------------------

    def trade(self, ticker, trade_side, order_type, quantity, price):
        risk_passed, msg = self.risk_check(ticker=ticker, quantity=quantity, trade_side=trade_side, price=price)
        if not risk_passed:
            return 0, f'Risk check failed: {msg}'

        trade_url = self._hook_ip + '/order/place'
        params = {'ticker': ticker, 'trade_side': trade_side, 'order_type': order_type, 'quantity': quantity,
                  'price': price, 'trade_environment': self._trading_environment}

        result = requests.post(trade_url, data=params).json()
        if result['ret_code'] == 1:
            order_id = result['return']['order_id']
            self._mq_socket.subscribe(f'{self._hook_name}.ORDER_UPDATE.{order_id}')
            # TODO: careful auto-type conversion
            df = pd.read_json(result['return']['content'], dtype={'order_id': str})
            df = df.rename(columns={'code': 'ticker'})
            self.update_positions(df=df)
            return 1, f'Placed order: {order_type} {quantity} qty of {ticker} @ {price}'
        else:
            return 0, f'Failed to place trade: {result["return"]["content"]}'

    def buy_market(self, ticker, quantity):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='BUY', order_type='MARKET', price=0.0)

    def sell_market(self, ticker, quantity):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='SELL', order_type='MARKET', price=0.0)

    def buy_limit(self, ticker, quantity, price):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='BUY', order_type='NORMAL', price=price)

    def sell_limit(self, ticker, quantity, price):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='SELL', order_type='NORMAL', price=price)

    def risk_check(self, ticker, quantity, trade_side, price):
        trade_sign = -1 if trade_side in ('BUY', 'BUY_BACK') else 1
        price = self.positions.loc[ticker]['price'] if price == 0.0 else price
        exp_cash_change = price * quantity * trade_sign
        self._current_margin = sum(abs(self.positions['market_value']))

        if self._current_cash + exp_cash_change < 0:
            return 0, f'Not enough cash, current cash:{self._current_cash} , required cash:{-exp_cash_change}'

        original_margin = abs(self.positions.loc[ticker]['market_value'])
        new_margin = abs(price * (self.positions.loc[ticker]['quantity'] + -1 * trade_sign * quantity))
        post_trade_margin = self._current_margin + new_margin - original_margin
        if post_trade_margin > self._margin:
            return 0, f'Not enough margin, current max margin:{self._margin} , required margin:{post_trade_margin}'

        if quantity % self.ticker_lot_size[ticker] != 0:
            return 0, f'Lot size is invalid, should be multiple of {self.ticker_lot_size[ticker]} but got {quantity}'

        return 1, 'Risk check passed'

    # ------------------------------------------------ [ Get infos ] ------------------------------------------
    def get_qty(self, ticker):
        return self.positions.loc[ticker]['quantity']

    def get_mv(self, ticker):
        return self.positions.loc[ticker]['market_value']

    def get_price(self, ticker):
        return self.positions.loc[ticker]['price']

    def get_lot_size(self, ticker):
        return self.ticker_lot_size[ticker]

    @property
    def cash(self):
        return self._current_cash

    @property
    def margin(self):
        return self._current_margin

    @property
    def available_cash(self):
        return self._current_cash

    @property
    def available_margin(self):
        self._current_margin = sum(abs(self.positions['market_value']))
        return self._margin - self._current_margin

    # ------------------------------------------------ [ Webapp ] ------------------------------------------

    async def get_attributes(self, request):
        return_attributes = dict()
        restricted_attr = (
            '_trading_universe', 'bars_no', 'logger', '_trading_environment', '_failed_tickers', '_datatypes',
            '_txn_cost', '_total_txn_cost',
            '_initial_capital', '_margin', '_running', '_current_cash', '_current_margin', 'pending_orders',
            'completed_orders', 'positions', 'slippage', 'ticker_lot_size', 'record', '_ip', '_mq_ip', '_hook_ip',
            '_zmq_context', '_mq_socket', '_topics', '_hook_name', 'cache_path', 'cache', '_max_cache',
            '_ticker_max_cache',
            '_cache_retain_ratio', 'cache_pickle_no_map', 'initialized_date', '_sanic', '_sanic_host', '_sanic_port',
            '_initialized', 'last_candlestick_time')
        for name, value in self.__dict__.items():
            if (type(value) in (list, str, float, int)) and (name not in restricted_attr):
                return_attributes[name] = value
        return response.json({'ret_code': 1, 'return': {'content': return_attributes}})

    async def get_summary(self, request):
        portfolio_value = sum(self.positions['market_value']) + self._current_cash
        trades = self.completed_orders.shape[0]
        days_since_deployment = max(int((datetime.datetime.today() - self.initialized_date).days), 1)
        self._current_margin = sum(abs(self.positions['market_value']))

        return response.json({'ret_code': 1, 'return': {'content': {'name': self.name,
                                                                    'benchmark': self.benchmark,
                                                                    'status': 'Running' if self._running else 'Paused',
                                                                    'initial_capital': self._initial_capital,
                                                                    'ip': self._ip,
                                                                    'pv': portfolio_value,
                                                                    'cash': self._current_cash,
                                                                    'margin': self._current_margin,
                                                                    'n_trades': trades,
                                                                    'txn_cost_total': self._total_txn_cost,
                                                                    'initialized_date': self.initialized_date.strftime(
                                                                        '%Y-%m-%d'),
                                                                    'days_since_deployment': days_since_deployment, }}})

    async def get_records(self, request):
        start_date = request.args.get('start_date')
        rows = request.args.get('rows')
        if rows is not None:
            record = self.record.iloc[-int(rows):]
        else:
            record = self.record

        if start_date is not None:
            record = record.loc[self.record.index >= start_date]

        pv = record['PV'].reset_index()
        pv.columns = ['x', 'y']
        pv = pv.to_dict('records')

        ev = record['EV'].reset_index()
        ev.columns = ['x', 'y']
        ev = ev.to_dict('records')

        cash = record['Cash'].reset_index()
        cash.columns = ['x', 'y']
        cash = cash.to_dict('records')

        margin = record['Margin'].reset_index()
        margin.columns = ['x', 'y']
        margin = margin.to_dict('records')

        return response.json({'ret_code': 1, 'return': {'content': {'PV': pv,
                                                                    'EV': ev,
                                                                    'Cash': cash,
                                                                    'Margin': margin}}})

    async def get_positions(self, request):
        positions = self.positions.reset_index()
        positions.columns = ['ticker', 'price', 'quantity', 'market_value']
        positions = positions.loc[abs(positions['market_value']) > 0]
        return response.json({'ret_code': 1, 'return': {'content': {'positions': positions.to_dict('records')}}})

    async def get_pending_orders(self, request):
        start_date = request.args.get('start_date')

        if len(self.pending_orders) > 0:
            pending_orders = pd.concat(self.pending_orders.values(), axis=1)

            if start_date is not None:
                pending_orders = pending_orders.loc[pending_orders['updated_time'] >= start_date]
        else:
            pending_orders = pd.DataFrame()
        return response.json(
            {'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}}, default=str)

    async def get_completed_orders(self, request):
        start_date = request.args.get('start_date')

        if self.completed_orders.shape[0] > 0:
            if start_date is not None:
                completed_orders = self.completed_orders.loc[self.completed_orders['updated_time'] >= start_date]
            else:
                completed_orders = self.completed_orders
        else:
            completed_orders = pd.DataFrame()
        return response.json(
            {'ret_code': 1, 'return': {'content': {'completed_orders': completed_orders.to_dict('records')}}})

    def subscribe_tickers(self, tickers):
        if len(tickers) > 0:
            self._trading_universe = list(set(self._trading_universe).union(tickers))
            succeed, failed = self.load_ticker_lot_size(tickers=tickers)

            self.load_all_cache(tickers=succeed)
            new_topics = self.add_new_topics(tickers=succeed)

            for ticker in succeed:
                self.positions.loc[ticker] = [0.0, 0.0, 0.0]

            for topic in new_topics:
                self._mq_socket.subscribe(topic)
                self.logger.debug(f'ZMQ subscribed to {topic}')

            for failed_ticker in failed:
                self.logger.debug(f'Failed to subscribe {failed_ticker}')

    async def web_subscribe_tickers(self, request):
        tickers = eval(request.args.get('tickers'))
        new_tickers = list(set(tickers).difference(self._trading_universe))

        self.subscribe_tickers(tickers=new_tickers)
        return response.json({'s': 's'})
        # TODO: return stuffs

    async def unsubscribe_ticker(self, request):
        tickers = eval(request.args.get('tickers'))
        tickers = list(set(tickers).intersection(self._trading_universe))

        new_topics = self.add_new_topics(tickers=tickers)
        for topic in new_topics:
            self._mq_socket.unsubscribe(topic)
        self._topics = list(set(self._topics).difference(new_topics))
        self._trading_universe = list(set(self._trading_universe).difference(tickers))
        # TODO: return stuffs

    async def pause(self, request):
        self._running = False
        # TODO: return stuffs

    async def resume(self, request):
        if self._initialized:
            self._running = True
        # TODO: return stuffs

    def app_add_route(self, app):
        app.add_route(self.get_records, '/curves', methods=['GET'])
        app.add_route(self.get_positions, '/positions', methods=['GET'])
        app.add_route(self.get_pending_orders, '/pending', methods=['GET'])
        app.add_route(self.get_completed_orders, '/completed', methods=['GET'])
        app.add_route(self.get_attributes, '/attributes', methods=['GET'])
        app.add_route(self.get_summary, '/summary', methods=['GET'])
        app.add_route(self.web_subscribe_tickers, '/subscribe', methods=['POST'])
        app.add_route(self.unsubscribe_ticker, '/unsubscribe', methods=['POST'])
        app.add_route(self.pause, '/pause', methods=['GET', 'SET', 'POST'])
        app.add_route(self.resume, '/resume', methods=['GET', 'SET', 'POST'])


class CandlestickStrategy(BaseAlgo):
    def __init__(self, name: str, bars_no: int, benchmark: str = 'HSI'):
        super().__init__(name=name, benchmark=benchmark)
        self.last_candlestick_time = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
        self.bars_no = bars_no

    def determine_trigger(self, datatype, ticker, df):
        # TODO: improve
        if 'K_' in datatype:
            datetime = df['datetime'].iloc[-1]
            last_df = self.last_candlestick_time[ticker][datatype]
            trigger_strat = (last_df is not None) and (datetime != last_df['datetime'].iloc[-1])
            self.last_candlestick_time[ticker][datatype] = df
            return trigger_strat, (
                datatype, ticker, self.get_data_rows(datatype=datatype, ticker=ticker, n_rows=self.bars_no + 1)[:-1])
        else:
            return True, (datatype, ticker, df)


class Backtest(BaseAlgo):
    def __init__(self, name: str, bars_no: int, benchmark: str = 'HSI'):
        super().__init__(name=name, benchmark=benchmark)
        self.bars_no = bars_no
        self._spread = 0
        self._order_queue = None
        self._cur_candlestick_datetime = None

    def determine_trigger(self, datatype, ticker, df):
        if 'K_' in datatype:
            return True, (
                datatype, ticker, self.get_data_rows(datatype=datatype, ticker=ticker, n_rows=self.bars_no))
        else:
            return True, (datatype, ticker, df)

    def initialize(self, initial_capital: float, margin: float, mq_ip: str,
                   hook_ip: str, hook_name: str, trading_environment: str,
                   trading_universe: list, datatypes: list,
                   txn_cost: float = 30, max_cache: int = 50000, ticker_max_cache: int = 3000,
                   cache_retain_ratio: float = 0.3, test_mq_con=False, spread: float = 0.2 / 100):
        # self.__init__(name=self.name, bars_no=self.bars_no, benchmark=self.benchmark)
        super().initialize(initial_capital=initial_capital, margin=margin, mq_ip=mq_ip, hook_ip=hook_ip,
                           hook_name=hook_name, trading_environment=trading_environment,
                           trading_universe=trading_universe,
                           txn_cost=txn_cost, max_cache=max_cache, ticker_max_cache=ticker_max_cache,
                           cache_retain_ratio=cache_retain_ratio, test_mq_con=test_mq_con, spread=spread,
                           datatypes=datatypes)
        self._spread = spread
        self._order_queue = list()

    def backtest(self, start_date, end_date):
        if not self._initialized:
            self.logger.info('Algo not initialized')
            return
        elif self._trading_environment != 'BACKTEST':
            self.logger.info('Environment is not BACKTEST')
            return

        self._running = True
        self.logger.info(f'Backtesting Starts...')

        # No need to load ticker cache
        succeed, failed = self.load_ticker_lot_size(tickers=self._trading_universe)
        for ticker in succeed:
            self.positions.loc[ticker] = [0.0, 0.0, 0.0]

        self.logger.info(f'Loading Date from MySQL DB...')
        backtest_df = pd.DataFrame()
        for ticker in self._trading_universe:
            for datatype in self._datatypes:
                ret_code, df = self.download_historical(ticker=ticker, datatype=datatype, start_date=start_date,
                                                        end_date=end_date)
                # TODO: different bars_no for different datatype
                filler = df.iloc[:self.bars_no]
                self.add_cache(datatype=datatype, df=filler)
                df = df.iloc[self.bars_no:]
                if ret_code != 1 or df.shape[0] == 0:
                    self.logger.error(
                        f'Failed to download data {datatype} {ticker} from Hook, please ensure data is in MySQL Db')
                else:
                    df['datatype'] = datatype
                    backtest_df = backtest_df.append(df)

        backtest_df = backtest_df.sort_values(by=['datetime', 'datatype', 'ticker'], ascending=True)

        self.logger.info(f'Loaded Data, backtesting starts...')

        async def _backtest():
            self._order_queue = list()
            # For progress bar
            last_percent = 0

            for i in range(backtest_df.shape[0]):
                # trigger orderUpdate first
                if len(self._order_queue) != 0:
                    for order_no in range(len(self._order_queue)):
                        order_update_df = self._order_queue.pop()
                        self.update_positions(df=order_update_df)
                        await self.on_order_update(order_id=order_update_df['order_id'].iloc[-1], df=order_update_df)


                # Progress Bar
                cur_percent = int(i / backtest_df.shape[0] * 100)
                if cur_percent != last_percent:
                    print(f'Progress: |{cur_percent * "#"}{(100 - cur_percent) * " "}| {i}/{backtest_df.shape[0]}')
                    last_percent = cur_percent

                # log performance
                tmp_df = backtest_df.iloc[i:i + 1]
                cur_candlestick_datetime = tmp_df['datetime'].iloc[-1]
                self._cur_candlestick_datetime = cur_candlestick_datetime
                if i > 0:
                    last_date = backtest_df.iloc[i - 1:i]['datetime'].iloc[-1].date()
                    if cur_candlestick_datetime.date() != last_date:
                        self.log(date=last_date)
                datatype = tmp_df['datatype'].iloc[-1]
                ticker = tmp_df['ticker'].iloc[-1]

                self.update_prices(datatype=datatype, df=tmp_df)
                self.add_cache(datatype=datatype, df=tmp_df)
                trigger_strat, (tgr_dtype, tgr_ticker, tgr_df) = self.determine_trigger(datatype=datatype,
                                                                                        ticker=ticker, df=tmp_df)
                if trigger_strat:
                    await self.trigger_strat(datatype=tgr_dtype, ticker=tgr_ticker, df=tgr_df)

            self.logger.info('Backtesting Completed! Call report() method to see backtesting result!')

        asyncio.run(_backtest())

    def trade(self, ticker, trade_side, order_type, quantity, price):
        risk_passed, msg = self.risk_check(ticker=ticker, quantity=quantity, trade_side=trade_side, price=price)
        if not risk_passed:
            return 0, f'Risk check failed: {msg}'

        spread_ajust_sign = 1 if 'BUY' in trade_side else -1
        spread_adjusted_price = price * (1 + (spread_ajust_sign * self._spread))

        backtest_trade = {'order_id': hash(time.time()), 'ticker': ticker, 'price': price,
                          'trd_side': trade_side, 'order_status': 'FILLED_ALL',
                          'dealt_avg_price': spread_adjusted_price,
                          'dealt_qty': quantity}
        order_update_df = pd.DataFrame(backtest_trade, index=[0])
        order_update_df['created_time'] = self._cur_candlestick_datetime
        self._order_queue.append(order_update_df)
        return 1, pd.DataFrame(backtest_trade, index=[0])


if __name__ == '__main__':
    pass
