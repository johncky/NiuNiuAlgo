from abc import ABC, abstractmethod
from FutuAlgo import logger
import zmq, pickle, asyncio, zmq.asyncio, time, requests, itertools, collections, random
from sanic import Sanic, response
import pandas as pd
from config import *
from tools import *


class BaseAlgo(ABC):
    def __init__(self, name: str, log_path='.', benchmark: str = 'HSI'):
        """ Define attributes with null values """
        # settings
        self.name = name
        self.benchmark = benchmark

        self._logger = logger.RootLogger(root_name=self.name, file_path=log_path)

        self._trading_environment = ''
        self._trading_universe = None
        self._failed_tickers = None
        self._datatypes = None
        self._txn_cost = None
        self._total_txn_cost = 0
        self._initial_capital = 0.0

        # current status
        self._running = False
        self._current_cash = 0.0

        # Info
        self._pending_orders = None
        self._completed_orders = None
        self._ticker_lot_size = None
        self.positions = None
        self.slippage = None
        self.records = None

        # IPs
        self._ip = ''
        self._mq_ip = ''
        self._hook_ip = ''
        self._zmq_context = None
        self._mq_socket = None
        self._topics = None
        self._hook_name = None

        # Cache
        self._cache = None
        self._cache_rows = 0
        self._prefill_period = None

        # Web
        self._sanic = None
        self._sanic_host = None
        self._sanic_port = None

        self._initialized_date = None
        self._initialized = False

    def initialize(self, initial_capital: float, mq_ip: str,
                   hook_ip: str, trading_environment: str,
                   trading_universe: list, datatypes: list,
                   txn_cost: float = 30, cache_rows: int = 3000,
                   test_mq_con=True, hook_name: str = 'FUTU', prefill_period='1Y', **kwargs):
        """ Initialize attributes and test connections with FutuHook ZMQ and Web API """

        assert trading_environment in (
        'BACKTEST', 'SIMULATE', 'REAL'), f'Invalid trading environment {trading_environment}'
        assert initial_capital > 0, 'Initial Capital cannot be 0'
        assert cache_rows > 1, 'No of cached data must be > 0 rows'

        try:
            valid_datatypes = list(set(datatypes).intersection(supported_dtypes))
            self._trading_environment = trading_environment
            self._trading_universe = trading_universe

            self._datatypes = valid_datatypes
            self._txn_cost = txn_cost
            self._total_txn_cost = 0

            self._pending_orders = dict()
            self.records = pd.DataFrame(columns=['PV', 'EV', 'Cash'])
            self._completed_orders = pd.DataFrame(columns=['order_id'])
            self.positions = pd.DataFrame(columns=['price', 'quantity', 'market_value'])
            self.slippage = pd.DataFrame(columns=['exp_price', 'dealt_price', 'dealt_qty', 'total_slippage'])

            self._initial_capital = initial_capital
            self._mq_ip = mq_ip
            self._hook_ip = hook_ip
            self._hook_name = hook_name
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
                        raise Exception(f'Failed to connect to ZMQ, please check : {self._mq_ip}')
                    self._logger.debug(f'Test Connection with ZMQ {self._mq_ip} is Successful!')
                    self._logger.debug(f'Test Connection with ZMQ {hello_mq_ip} is Successful!')

                except zmq.error.Again:
                    raise Exception(f'Failed to connect to ZMQ, please check {self._mq_ip}')
                finally:
                    test_context.destroy()

            # Test Connection with Hook
            try:
                requests.get(self._hook_ip + '/subscriptions').json()
                self._logger.debug(f'Test Connection with FutuHook IP f{self._hook_ip} is Successful!')
            except requests.ConnectionError:
                raise Exception(f'Connection with FutuHook failed, please check: {self._hook_ip}')

            self._ticker_lot_size = dict()
            self._failed_tickers = list()
            self._topics = list()

            # Cache
            self._cache = collections.defaultdict(lambda: collections.defaultdict(lambda: pd.DataFrame()))
            self._cache_rows = cache_rows
            self._prefill_period = prefill_period

            self._initialized_date = datetime.datetime.today()
            self._running = False

            self._initialized = True
            self._logger.debug('Initialized sucessfully.')

        except Exception as e:
            self._initialized = False
            self._logger.error(f'Failed to initialize algo, reason: {str(e)}')

    async def record_daily_performance(self):
        while True:
            self.log()
            await asyncio.sleep(60 * 60 * 24 - time.time() % 60 * 60 * 24)

    def log(self, overwrite_date=None):
        ev = sum(self.positions['market_value'])
        pv = ev + self._current_cash
        d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if overwrite_date is None else overwrite_date
        self.records.loc[d] = [pv, ev, self._current_cash]

    async def main(self):

        self._zmq_context = zmq.asyncio.Context()
        self._mq_socket = self._zmq_context.socket(zmq.SUB)
        self._mq_socket.connect(self._mq_ip)
        self.subscribe_tickers(tickers=self._trading_universe, prefill_period=self._prefill_period)

        self._running = True
        self._logger.debug(f'Algo {self.name} running successfully!')

        while True:

            try:
                topic, bin_df = await self._mq_socket.recv_multipart()
                if not self._running:
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
                    self.add_cache(datatype=datatype, df=df, ticker=key)
                    trigger_strat, (tgr_dtype, tgr_ticker, tgr_df) = self.determine_trigger(datatype=datatype,
                                                                                            ticker=key, df=df)
                    if trigger_strat:
                        await self.trigger_strat(datatype=tgr_dtype, ticker=tgr_ticker, df=tgr_df)
            except Exception as e:
                self._running = False
                self._logger.error(f'Exception occur, Algo stopped due to {str(e)}')

    def determine_trigger(self, datatype, ticker, df):
        """ logic that determine when to trigger events, return True in first element when trigger """
        return True, (datatype, ticker, df)

    async def trigger_strat(self, datatype, ticker, df):
        """ route triggers to methods """
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
            self._logger.debug('Algo not initialized')
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
                tasks.append(self.record_daily_performance())
                await asyncio.gather(*tasks)

            loop.create_task(_run())
            loop.run_forever()

    # ------------------------------------------------ [ Position ] ------------------------------------------
    def update_positions(self, df):
        """ Update position and cash on receiving updates from broker """
        trd_side = 1 if df['trd_side'].iloc[0].upper() in ('BUY', 'BUY_BACK') else -1
        dealt_qty = df['dealt_qty'].iloc[0] * trd_side
        avg_price = df['dealt_avg_price'].iloc[0]
        order_id = df['order_id'].iloc[0]
        ticker = df['ticker'].iloc[0]
        order_status = df['order_status'].iloc[0]

        in_pending = False
        in_completed = False
        if order_id in self._pending_orders.keys():
            in_pending = True
            last_order_update_df = self._pending_orders[order_id]

            last_qty = last_order_update_df['dealt_qty'].iloc[0] * trd_side
            last_avg_price = last_order_update_df['dealt_avg_price'].iloc[0]

            cash_change = -(dealt_qty * avg_price - last_qty * last_avg_price)
            qty_change = dealt_qty - last_qty

        else:
            if order_id not in self._completed_orders['order_id']:
                cash_change = - dealt_qty * avg_price
                qty_change = dealt_qty
            else:
                in_completed = True
                cash_change = 0
                qty_change = 0

        if order_status in ('SUBMIT_FAILED', 'FILLED_ALL', 'CANCELLED_PART', 'CANCELLED_ALL', 'FAILED', 'DELETED'):
            if not in_completed:
                self._completed_orders = self._completed_orders.append(df)

                if order_status in ('FILLED_ALL', 'CANCELLED_PART'):
                    # update slippage
                    self.slippage.loc[order_id] = [0, 0, 0, 0]
                    exp_price = self.positions.loc[ticker]['price'] if df['price'].iloc[0] == 0.0 else df['price'].iloc[
                        0]
                    self.slippage.loc[order_id] = [exp_price, avg_price, dealt_qty, (avg_price - exp_price) * dealt_qty]

                    # Txn cost
                    self._total_txn_cost += self._txn_cost
                    cash_change -= self._txn_cost

                if in_pending:
                    del self._pending_orders[order_id]

        else:
            self._pending_orders[order_id] = df

        # update positions and snapshot
        latest_price = self.positions.loc[ticker]['price']
        existing_qty = self.positions.loc[ticker]['quantity']

        new_qty = existing_qty + qty_change
        self.positions.loc[ticker] = [latest_price, new_qty, new_qty * latest_price]
        self._current_cash += cash_change

    def update_prices(self, datatype, df):
        """ Update price for valuation of positions """
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
    def add_cache(self, datatype, ticker, df):
        self._cache[datatype][ticker] = self._cache[datatype][ticker].append(df).drop_duplicates(
            subset=['datetime', 'ticker'], keep='last')
        self._cache[datatype][ticker] = self._cache[datatype][ticker].iloc[-self._cache_rows:]

    def get_data(self, datatype, ticker: str, start_date: datetime.datetime = None, n_rows: int = None, sort_drop=True):
        df = self._cache[datatype][ticker]
        if start_date:
            df = df.loc[df['datetime'] >= start_date]
        if n_rows:
            df = df.iloc[-n_rows:]
        if sort_drop:
            df = df.drop_duplicates(['datetime', 'ticker'], keep='last').sort_values(['datetime'])
        return df

    def download_historical(self, ticker, datatype, start_date=None, end_date=None, from_exchange=False):
        params = {'ticker': ticker, 'datatype': datatype, 'start_date': start_date, 'end_date': end_date,
                  'from_exchange': from_exchange}
        result = requests.get(self._hook_ip + '/historicals', params=params).json()
        if result['ret_code'] == 1:
            df = pd.read_json(result['return']['content'])
            return 1, df
        else:
            return 0, result['return']['content']

    def download_ticker_data(self, ticker, datatype, start_date, from_exchange=False):
        ret_code, df = self.download_historical(ticker=ticker, datatype=datatype, start_date=start_date,
                                                from_exchange=from_exchange)

        if ret_code == 1:
            self.add_cache(datatype=datatype, ticker=ticker, df=df)
        else:
            raise Exception(f'Failed to download historical data from Hook due to {df}')

    def download_all_data(self, start_date, tickers=None):

        tickers = self._trading_universe if tickers is None else tickers
        for dtype in self._datatypes:
            for ticker in tickers:
                self.download_ticker_data(ticker=ticker, datatype=dtype, start_date=start_date)

    def download_ticker_lot_size(self, tickers):
        params = {'tickers': str(tickers)}

        result = requests.get(self._hook_ip + '/order/lot_size', params=params).json()
        if result['ret_code'] == 1:
            lot_size_df = pd.read_json(result['return']['content'])
            failed = list(lot_size_df.loc[lot_size_df['lot_size'] == 0].index)
            succeed = list(lot_size_df.loc[lot_size_df['lot_size'] > 0].index)
            for ticker in succeed:
                self._ticker_lot_size[ticker] = lot_size_df.loc[ticker]['lot_size']
            self._trading_universe = list(set(self._trading_universe).difference(failed))
            self._failed_tickers = list(set(self._failed_tickers).union(failed))
            return succeed, failed
        else:
            raise Exception(f'Failed to request lot size due to {result["return"]["content"]}')

    def add_zmq_topics(self, tickers):
        tmp = [f'{self._hook_name}.{x[0]}.{x[1]}' for x in itertools.product(self._datatypes, tickers)]
        self._topics = list(set(self._topics).union(tmp))
        return tmp

    # ------------------------------------------------ [ Trade ] ------------------------------------------

    def trade(self, ticker, trade_side, order_type, quantity, price):
        risk_passed, msg = self.pre_trade_check(ticker=ticker, quantity=quantity, trade_side=trade_side, price=price)
        if not risk_passed:
            msg = f'Risk check failed:"{order_type} {quantity} qty of {ticker} @ {price}" due to {msg}'
            self._logger.info(msg)
            return 0, msg

        trade_url = self._hook_ip + '/order/place'
        params = {'ticker': ticker, 'trade_side': trade_side, 'order_type': order_type, 'quantity': int(quantity),
                  'price': price, 'trade_environment': self._trading_environment}

        result = requests.post(trade_url, data=params).json()
        if result['ret_code'] == 1:
            order_id = result['return']['order_id']
            self._mq_socket.subscribe(f'{self._hook_name}.ORDER_UPDATE.{order_id}')
            df = pd.read_json(result['return']['content'], dtype={'order_id': str})
            # df = df.rename(columns={'code': 'ticker'})
            self.update_positions(df=df)
            msg = f'Placed order: {order_type} {quantity} qty of {ticker} @ {price}'
            self._logger.info(msg)
            return 1, msg
        else:
            msg = f'Failed to place trade due to {result["return"]["content"]}'
            self._logger.info(msg)
            return 0, msg

    def buy_market(self, ticker, quantity):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='BUY', order_type='MARKET', price=0.0)

    def sell_market(self, ticker, quantity):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='SELL', order_type='MARKET', price=0.0)

    def buy_limit(self, ticker, quantity, price):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='BUY', order_type='NORMAL', price=price)

    def sell_limit(self, ticker, quantity, price):
        return self.trade(ticker=ticker, quantity=quantity, trade_side='SELL', order_type='NORMAL', price=price)

    def pre_trade_check(self, ticker, quantity, trade_side, price):
        trade_sign = -1 if trade_side == 'BUY' else 1
        price = self.positions.loc[ticker]['price'] if price == 0.0 else price
        exp_cash_change = price * quantity * trade_sign

        if quantity <= 0:
            return 0, f'Quantity cannot be <= 0! '

        if self._current_cash + exp_cash_change < 0:
            return 0, f'Not enough cash, current cash:{self._current_cash} , required cash:{-exp_cash_change}'

        if quantity % self._ticker_lot_size[ticker] != 0:
            return 0, f'Lot size is invalid, should be multiple of {self._ticker_lot_size[ticker]} but got {quantity}'

        return 1, 'Risk check passed'

    # ------------------------------------------------ [ Get infos ] ------------------------------------------
    def get_current_qty(self, ticker):
        return self.positions.loc[ticker]['quantity']

    def get_current_market_value(self, ticker):
        return self.positions.loc[ticker]['market_value']

    def get_latest_price(self, ticker):
        return self.positions.loc[ticker]['price']

    def get_lot_size(self, ticker):
        return self._ticker_lot_size[ticker]

    def calc_max_buy_qty(self, ticker, cash=None, adjust_limit=1.03):
        cash = self._current_cash if cash is None else cash
        lot_size = self.get_lot_size(ticker)
        one_hand_size = self.get_lot_size(ticker) * self.get_latest_price(ticker) * adjust_limit
        if cash >= one_hand_size:
            max_qty_by_cash = int((cash - cash % one_hand_size) / one_hand_size) * lot_size
            return max_qty_by_cash
        else:
            return 0

    @property
    def cash(self):
        return self._current_cash

    @property
    def current_cash(self):
        return self._current_cash

    # ------------------------------------------------ [ Webapp ] ------------------------------------------

    async def get_attributes(self, request):
        return_attributes = dict()
        restricted_attr = (
            '_trading_universe', 'logger', '_trading_environment', '_failed_tickers', '_datatypes',
            '_txn_cost', '_total_txn_cost',
            '_initial_capital', '_running', '_current_cash', '_pending_orders',
            '_completed_orders', 'positions', 'slippage', '_ticker_lot_size', 'record', '_ip', '_mq_ip', '_hook_ip',
            '_zmq_context', '_mq_socket', '_topics', '_hook_name', '_cache',
            '_cache_rows', '_initialized_date', '_sanic', '_sanic_host', '_sanic_port', '_prefill_period',
            '_initialized', '_last_update', '_bars_window')
        for name, value in self.__dict__.items():
            if (type(value) in (list, str, float, int)) and (name not in restricted_attr):
                return_attributes[name] = value
        return response.json({'ret_code': 1, 'return': {'content': return_attributes}})

    async def get_summary(self, request):
        portfolio_value = sum(self.positions['market_value']) + self._current_cash
        trades = self._completed_orders.shape[0]
        days_since_deployment = max(int((datetime.datetime.today() - self._initialized_date).days), 1)

        return response.json({'ret_code': 1, 'return': {'content': {'name': self.name,
                                                                    'benchmark': self.benchmark,
                                                                    'status': 'Running' if self._running else 'Paused',
                                                                    'initial_capital': self._initial_capital,
                                                                    'ip': self._ip,
                                                                    'pv': portfolio_value,
                                                                    'cash': self._current_cash,
                                                                    'n_trades': trades,
                                                                    'txn_cost_total': self._total_txn_cost,
                                                                    'initialized_date': self._initialized_date.strftime(
                                                                        '%Y-%m-%d'),
                                                                    'days_since_deployment': days_since_deployment, }}})

    async def get_records(self, request):
        start_date = request.args.get('start_date')
        rows = request.args.get('rows')

        record = self.records
        if rows is not None:
            record = record.iloc[-int(rows):]

        if start_date is not None:
            record = record.loc[self.records.index >= start_date]

        pv = record['PV'].reset_index()
        pv.columns = ['x', 'y']
        pv = pv.to_dict('records')

        ev = record['EV'].reset_index()
        ev.columns = ['x', 'y']
        ev = ev.to_dict('records')

        cash = record['Cash'].reset_index()
        cash.columns = ['x', 'y']
        cash = cash.to_dict('records')

        return response.json({'ret_code': 1, 'return': {'content': {'PV': pv,
                                                                    'EV': ev,
                                                                    'Cash': cash}}})

    async def get_positions(self, request):
        positions = self.positions.reset_index()
        positions.columns = ['ticker', 'price', 'quantity', 'market_value']
        positions = positions.loc[abs(positions['market_value']) > 0]
        return response.json({'ret_code': 1, 'return': {'content': {'positions': positions.to_dict('records')}}})

    async def get_pending_orders(self, request):
        start_date = request.args.get('start_date')

        if len(self._pending_orders) > 0:
            pending_orders = pd.concat(self._pending_orders.values(), axis=1)

            if start_date is not None:
                pending_orders = pending_orders.loc[pending_orders['updated_time'] >= start_date]
        else:
            pending_orders = pd.DataFrame()
        # return response.json({'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}}, default=str)
        return response.json(
            {'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}})

    async def get_completed_orders(self, request):
        start_date = request.args.get('start_date')

        if self._completed_orders.shape[0] > 0:
            if start_date is not None:
                completed_orders = self._completed_orders.loc[self._completed_orders['updated_time'] >= start_date]
            else:
                completed_orders = self._completed_orders
        else:
            completed_orders = pd.DataFrame()
        return response.json(
            {'ret_code': 1, 'return': {'content': {'completed_orders': completed_orders.to_dict('records')}}})

    def subscribe_tickers(self, tickers, prefill_period):
        if len(tickers) > 0:
            tickers = pd.unique(tickers).tolist()
            self._trading_universe = list(set(self._trading_universe).union(tickers))
            succeed, failed = self.download_ticker_lot_size(tickers=tickers)

            self.download_all_data(tickers=succeed, start_date=period_to_start_date(prefill_period))
            new_topics = self.add_zmq_topics(tickers=succeed)

            for ticker in succeed:
                self.positions.loc[ticker] = [0.0, 0.0, 0.0]

            for topic in new_topics:
                self._mq_socket.subscribe(topic)
                self._logger.debug(f'ZMQ subscribed to {topic}')

            for failed_ticker in failed:
                self._logger.debug(f'Failed to subscribe {failed_ticker}')

    async def web_subscribe_tickers(self, request):
        tickers = eval(request.args.get('tickers'))
        new_tickers = list(set(tickers).difference(self._trading_universe))

        self.subscribe_tickers(tickers=new_tickers, prefill_period=self._prefill_period)
        try:
            return response.json({'ret_code': 1, 'return': {'content': {'universe': list(self._trading_universe),
                                                                        'datatypes': list(self._datatypes)}}})
        except Exception as e:
            return response.json({'ret_code': 0, 'return': {'content': str(e)}})

    async def unsubscribe_ticker(self, request):
        tickers = eval(request.args.get('tickers'))
        tickers = list(set(tickers).intersection(self._trading_universe))

        new_topics = self.add_zmq_topics(tickers=tickers)
        for topic in new_topics:
            self._mq_socket.unsubscribe(topic)
        self._topics = list(set(self._topics).difference(new_topics))
        self._trading_universe = list(set(self._trading_universe).difference(tickers))
        try:
            return response.json({'ret_code': 1, 'return': {'content': {'universe': list(self._trading_universe),
                                                                        'datatypes': list(self._datatypes)}}})
        except Exception as e:
            return response.json({'ret_code': 0, 'return': {'content': str(e)}})

    async def pause(self, request):
        self._running = False
        return response.json({'ret_code': 1, 'return': {'content': {'running': self._running}}})

    async def resume(self, request):
        if self._initialized:
            self._running = True
        return response.json({'ret_code': 1, 'return': {'content': {'running': self._running}}})

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
    def __init__(self, name: str, bars_window: int, benchmark: str = 'HSI'):
        super().__init__(name=name, benchmark=benchmark)
        self._last_update = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
        self._bars_window = bars_window

    def determine_trigger(self, datatype, ticker, df):
        if 'K_' in datatype:
            datetime = df['datetime'].iloc[-1]
            last_df = self._last_update[ticker][datatype]
            trigger_strat = (last_df is not None) and (datetime != last_df['datetime'].iloc[-1])
            self._last_update[ticker][datatype] = df
            return trigger_strat, (
                datatype, ticker, self.get_data(datatype=datatype, ticker=ticker, n_rows=self._bars_window + 1)[:-1])
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
                datatype, ticker, self.get_data(datatype=datatype, ticker=ticker, n_rows=self.bars_no))
        else:
            return True, (datatype, ticker, df)

    def initialize(self, initial_capital: float,
                   hook_ip: str,
                   trading_universe: list, datatypes: list,
                   txn_cost: float = 30, cache_rows: int = 3000,
                   test_mq_con=False, spread: float = 0.2 / 100, **kwargs):

        super().initialize(initial_capital=initial_capital, mq_ip='', hook_ip=hook_ip,
                           trading_environment='BACKTEST',
                           trading_universe=trading_universe,
                           txn_cost=txn_cost, cache_rows=cache_rows,
                           test_mq_con=test_mq_con, spread=spread,
                           datatypes=datatypes)
        self._spread = spread
        self._order_queue = list()

    def backtest(self, start_date, end_date):
        if not self._initialized:
            self._logger.debug('Algo not initialized')
            return

        self._logger.debug(f'Backtesting Starts...')

        # No need to load ticker cache
        succeed, failed = self.download_ticker_lot_size(tickers=self._trading_universe)
        for ticker in succeed:
            self.positions.loc[ticker] = [0.0, 0.0, 0.0]

        self._logger.debug(f'Loading Date from MySQL DB...')
        backtest_df = pd.DataFrame()
        for tk in self._trading_universe:
            for dtype in self._datatypes:
                ret_code, df = self.download_historical(ticker=tk, datatype=dtype, start_date=start_date,
                                                        end_date=end_date)
                if ret_code != 1 or df.shape[0] == 0:
                    msg = f'Failed to download data {dtype} {tk} from Hook, please ensure data is in MySQL Db'
                    self._logger.error(msg)
                    raise Exception(msg)
                else:
                    df['datatype'] = dtype

                # TODO: different bars_no for different datatype
                filler = df.iloc[:self.bars_no]
                df = df.iloc[self.bars_no:]
                if df.shape[0] > 0:
                    backtest_df = backtest_df.append(df)
                    self.add_cache(datatype=dtype, df=filler, ticker=tk)
                    self._logger.debug(
                        f'Backtesting {tk} from {df["datetime"].iloc[0]}')
                else:
                    self._logger.warn(f'Not Enough bars to backtest {dtype}.{tk}')
                    continue

        backtest_df = backtest_df.sort_values(by=['datetime', 'datatype', 'ticker'], ascending=True)

        self._logger.debug(f'Loaded Data, backtesting starts...')

        async def _backtest():
            self._order_queue = list()
            # For progress bar
            last_percent = 0
            self.log(overwrite_date=backtest_df.iloc[0]['datetime'].date())
            for i in range(backtest_df.shape[0]):
                cur_df = backtest_df.iloc[i:i + 1]
                datatype = cur_df['datatype'].iloc[-1]
                ticker = cur_df['ticker'].iloc[-1]
                self._cur_candlestick_datetime = cur_df['datetime'].iloc[-1]

                # trigger orderUpdate first
                if len(self._order_queue) != 0:
                    tmp_order_queue = list()
                    for order_no in range(len(self._order_queue)):
                        action_type, data = self._order_queue.pop()
                        if action_type == 'UPDATE':
                            await self.on_order_update(order_id=data['order_id'].iloc[-1], df=data)
                        elif action_type == 'EXECUTE':
                            if datatype == data['datatype'] and ticker == data['ticker']:
                                self.trade(ticker=data['ticker'], trade_side=data['trade_side'],
                                           order_type='MARKET', quantity=data['quantity'],
                                           price=cur_df['open'].iloc[-1])
                            else:
                                tmp_order_queue.append((action_type, data))
                    self._order_queue = self._order_queue + tmp_order_queue

                # Progress Bar
                cur_percent = int(i / backtest_df.shape[0] * 100)
                if cur_percent != last_percent:
                    print(f'Progress: |{cur_percent * "#"}{(100 - cur_percent) * " "}| {i}/{backtest_df.shape[0]}')
                    last_percent = cur_percent

                # log performance
                if i > 0:
                    last_date = backtest_df.iloc[i - 1:i]['datetime'].iloc[0].date()
                    if self._cur_candlestick_datetime.date() != last_date:
                        self.log(overwrite_date=last_date)

                self.update_prices(datatype=datatype, df=cur_df)
                self.add_cache(datatype=datatype, df=cur_df, ticker=ticker)
                trigger_strat, (tgr_dtype, tgr_ticker, tgr_df) = self.determine_trigger(datatype=datatype,
                                                                                        ticker=ticker, df=cur_df)
                if trigger_strat:
                    await self.trigger_strat(datatype=tgr_dtype, ticker=tgr_ticker, df=tgr_df)

            self._logger.debug('Backtesting Completed! Call report() method to see backtesting result!')

        asyncio.run(_backtest())

    def trade(self, ticker, trade_side, order_type, quantity, price):
        risk_passed, msg = self.pre_trade_check(ticker=ticker, quantity=quantity, trade_side=trade_side, price=price)
        if not risk_passed:
            self._logger.warn(
                f'Risk check for order "{trade_side} {quantity} qty of {ticker} @ {price}" did not pass, reasons: {msg}')
            backtest_trade = {'order_id': hash(random.random()), 'ticker': ticker, 'price': 0,
                              'trd_side': trade_side, 'order_status': 'FAILED',
                              'dealt_avg_price': 0,
                              'dealt_qty': 0, 'created_time': 0, 'last_err_msg': f'Risk check failed: {msg}'}

            order_update_df = pd.DataFrame(backtest_trade, index=[0])
            self._order_queue.append(('UPDATE', order_update_df))
            return 0, f'Risk check failed: {msg}'

        spread_ajust_sign = 1 if 'BUY' in trade_side else -1
        order_datetime = self._cur_candlestick_datetime
        spread_adjusted_price = price * (1 + (spread_ajust_sign * self._spread))

        backtest_trade = {'order_id': hash(random.random()), 'ticker': ticker, 'price': price,
                          'trd_side': trade_side, 'order_status': 'FILLED_ALL',
                          'dealt_avg_price': spread_adjusted_price,
                          'dealt_qty': quantity, 'created_time': order_datetime}

        order_update_df = pd.DataFrame(backtest_trade, index=[0])
        self.update_positions(df=order_update_df)
        self._order_queue.append(('UPDATE', order_update_df))
        return 1, f'Placed order: {order_type} {quantity} qty of {ticker} @ {price}'

    def buy_market(self, ticker, quantity):
        # Buy at current close
        return self.trade(ticker=ticker, quantity=quantity, trade_side='BUY', order_type='MARKET',
                          price=self.get_latest_price(ticker))

    def sell_market(self, ticker, quantity):
        # Sell at current close
        return self.trade(ticker=ticker, quantity=quantity, trade_side='SELL', order_type='MARKET',
                          price=self.get_latest_price(ticker))

    def buy_limit(self, ticker, quantity, price):
        # Buy at current close
        return self.trade(ticker=ticker, quantity=quantity, trade_side='BUY', order_type='NORMAL',
                          price=self.get_latest_price(ticker))

    def sell_limit(self, ticker, quantity, price):
        # Sell at current close
        return self.trade(ticker=ticker, quantity=quantity, trade_side='SELL', order_type='NORMAL',
                          price=self.get_latest_price(ticker))

    def buy_next_open(self, datatype, ticker, quantity):
        # Buy at next open of that datatype
        self._order_queue.append(
            ('EXECUTE', {'datatype': datatype, 'ticker': ticker, 'quantity': quantity, 'trade_side': 'BUY'}))
        return 1, f'Buy {quantity} {ticker} @ next {datatype} open'

    def sell_next_open(self, datatype, ticker, quantity):
        # Sell at next open of that datatype
        self._order_queue.append(
            ('EXECUTE', {'datatype': datatype, 'ticker': ticker, 'quantity': quantity, 'trade_side': 'SELL'}))
        return 1, f'Sell {quantity} {ticker} @ next {datatype} open'

    # ------------------------------------------------ [ Report ] ------------------------------------------
    def plot_ticker_trades(self, datatype, ticker):
        orders_df = self._completed_orders.loc[self._completed_orders.ticker == ticker].rename(
            columns={'created_time': 'datetime'})
        ticker_df = self._cache[datatype][ticker]
        ticker_df = ticker_df.merge(orders_df[['datetime', 'trd_side']], how='left', on=['datetime'])
        ticker_df = ticker_df.fillna(0)
        ticker_df['buy_pt'] = [1 if 'BUY' in str(x) else None for x in ticker_df['trd_side']]
        ticker_df['sell_pt'] = [1 if 'SELL' in str(x) else None for x in ticker_df['trd_side']]
        ticker_df['buy_y'] = ticker_df['buy_pt'] * ticker_df['close']
        ticker_df['sell_y'] = ticker_df['sell_pt'] * ticker_df['close']
        ticker_df['x'] = ticker_df['datetime']
        from matplotlib import pyplot as plt
        plt.scatter(x=ticker_df['x'], y=ticker_df['buy_y'].values, marker='o', color='green', s=100)
        plt.scatter(x=ticker_df['x'], y=ticker_df['sell_y'].values, marker='o', color='red', s=100)
        plt.ylabel(f'{ticker} price')
        plt.plot(ticker_df['x'], ticker_df['close'])
        plt.title(f'{ticker} entry-exit points')

    def report(self, benchmark):
        import quantstats as qs
        import webbrowser

        PV = self.records['PV']
        PV.index = pd.to_datetime(PV.index)
        PV.index.name = 'datetime'
        PV = PV.resample('1D').last().fillna(method='ffill')
        html = f'{self.name}.html'
        qs.reports.html(PV, benchmark, output=html, title=f'{self.name}')
        webbrowser.open(html)


if __name__ == '__main__':
    pass
