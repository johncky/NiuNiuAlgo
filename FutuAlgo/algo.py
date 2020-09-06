from account import Account
from execution import Execution
from FutuAlgo import logger
from tools import *
import pickle
import asyncio
import time
import collections
import random
from sanic import Sanic
import pandas as pd
from strategy import Strategy
from data import Data



class Algo(Strategy):
    """ Class that ensemble Account, Strategy, Data and Web to run algo trading """
    def __init__(self, name: str, log_path='.', benchmark: str = 'HSI'):
        self.name = name
        self.benchmark = benchmark
        self._logger = logger.RootLogger(root_name=self.name, file_path=log_path)
        self._running = False

        self._ip = None
        self._sanic = None
        self._sanic_host = None
        self._sanic_port = None

        self._initialized_date = None
        self._initialized = False

        self._data = None
        self._account = None
        self._execution=None

    @try_expt(msg='Initialization Failed', pnt_original=True)
    def initialize(self, initial_capital: float, mq_ip: str,
                   hook_ip: str, trading_environment: str,
                   trading_universe: list, datatypes: list,
                   txn_cost: float = 30, cache_rows: int = 3000,
                   test_mq_con=True, hook_name: str = 'FUTU', prefill_period='1Y', **kwargs):
        assert trading_environment in ('BACKTEST', 'SIMULATE', 'REAL'), f'Invalid trading environment {trading_environment}'
        assert initial_capital > 0, 'Initial Capital cannot be 0'
        assert cache_rows > 1, 'No of cached data must be > 0 rows'
        self._account = Account(logger=self._logger, initial_capital=initial_capital, txn_cost=txn_cost)
        self._data = Data(mq_ip=mq_ip, logger=self._logger, hook_ip=hook_ip, trading_universe=trading_universe,
                          datatypes=datatypes, cache_rows=cache_rows, test_mq_con=test_mq_con, hook_name=hook_name,
                          prefill_period=prefill_period, add_pos_func=self._account.add_new_position)
        self._execution = Execution(account=self._account, data=self._data, trading_environment=trading_environment,
                                    logger=self._logger)

        self._initialized_date = datetime.datetime.today()
        self._running = False
        self._logger.debug('Initialized sucessfully.')
        self._initialized = True

    async def record_daily_performance(self):
        while True:
            self._account.log()
            await asyncio.sleep(60 * 60 * 24 - time.time() % 60 * 60 * 24)

    async def main(self):
        self._running = True
        self._data.start_sub()
        self._logger.debug(f'Algo {self.name} is running successfully!')

        while True:
            try:
                topic, bin_df = await self._data.receive_data()
                if not self._running:
                    continue

                topic_split = topic.decode('ascii').split('.')
                datatype = topic_split[1]
                key = '.'.join(topic_split[2:])
                df = pickle.loads(bin_df)
                if datatype == 'ORDER_UPDATE':
                    self._account.update_positions(df)
                    await self.on_order_update(order_id=key, df=df)
                else:
                    self._account.update_prices(datatype=datatype, df=df)
                    self._data.add_cache(datatype=datatype, df=df, ticker=key)
                    trigger_strat, (tgr_dtype, tgr_ticker, tgr_df) = self.determine_trigger(datatype=datatype,
                                                                                            ticker=key, df=df)
                    if trigger_strat:
                        await self.trigger_strat(datatype=tgr_dtype, ticker=tgr_ticker, df=tgr_df)
            except Exception as e:
                self._running = False
                self._logger.error(f'Exception occur, Algo stopped due to {str(e)}')

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

    # ------------------------------------------------ [ Get infos ] ------------------------------------------

    def get_current_qty(self, ticker):
        return self._account.get_current_qty(ticker=ticker)

    def get_current_market_value(self, ticker):
        return self._account.get_current_market_value(ticker=ticker)

    def get_latest_price(self, ticker):
        return self._account.get_latest_price(ticker=ticker)

    def get_lot_size(self, ticker):
        return self._data.get_lot_size(ticker=ticker)

    def calc_max_buy_qty(self, ticker, cash=None, adjust_limit=1.03):
        lot_size = self.get_lot_size(ticker=ticker)
        return self._account.calc_max_buy_qty(ticker=ticker, lot_size=lot_size, cash=cash, adjust_limit=adjust_limit)

    @property
    def cash(self):
        return self._account.cash

    @property
    def current_cash(self):
        return self._account.current_cash

    def get_data(self, datatype, ticker: str, start_date: datetime.datetime = None, n_rows: int = None, sort_drop=True):
        return self._data.get_data(datatype=datatype, ticker=ticker, start_date=start_date, n_rows=n_rows,
                                   sort_drop=sort_drop)

    # ------------------------------------------------ [ Webapp ] ------------------------------------------
    @web_expt()
    async def get_attributes(self, request):
        return_attributes = dict()
        for name, value in self.__dict__.items():
            if (type(value) in (list, str, float, int)) and (name[0] != '_'):
                return_attributes[name] = value
        return response.json({'ret_code': 1, 'return': {'content': return_attributes}})

    @web_expt()
    async def get_summary(self, request):
        days_since_deployment = max(int((datetime.datetime.today() - self._initialized_date).days), 1)

        return response.json({'ret_code': 1, 'return': {'content': {'name': self.name,
                                                                    'benchmark': self.benchmark,
                                                                    'status': 'Running' if self._running else 'Paused',
                                                                    'initial_capital': self._account.init_capital,
                                                                    'ip': self._ip,
                                                                    'pv': self._account.mv + self._account.current_cash,
                                                                    'cash': self.current_cash,
                                                                    'n_trades': self._account.n_trades,
                                                                    'txn_cost_total': self._account.total_txn_cost,
                                                                    'initialized_date': self._initialized_date.strftime(
                                                                        '%Y-%m-%d'),
                                                                    'days_since_deployment': days_since_deployment}}})

    @web_expt()
    async def get_records(self, request):
        start_date = request.args.get('start_date')
        rows = request.args.get('rows')

        record = self._account.records
        if rows is not None:
            record = record.iloc[-int(rows):]

        if start_date is not None:
            record = record.loc[record.index >= start_date]

        time_series = dict()
        for ts in ('PV', 'EV', 'Cash'):
            ts_series = record[ts].reset_index()
            ts_series.columns = ['x', 'y']
            ts_series = ts_series.to_dict('records')
            time_series[ts] = ts_series

        return response.json({'ret_code': 1, 'return': {'content': time_series}})

    @web_expt()
    async def get_positions(self, request):
        positions = self._account.positions.reset_index()
        positions.columns = ['ticker', 'price', 'quantity', 'market_value']
        positions = positions.loc[abs(positions['market_value']) > 0]
        return response.json({'ret_code': 1, 'return': {'content': {'positions': positions.to_dict('records')}}})

    @web_expt()
    async def get_pending_orders(self, request):
        start_date = request.args.get('start_date')
        pending_orders = self._account.pending_orders
        if len(pending_orders) > 0:
            pending_orders = pd.concat(pending_orders.values(), axis=1)

            if start_date is not None:
                pending_orders = pending_orders.loc[pending_orders['updated_time'] >= start_date]
        else:
            pending_orders = pd.DataFrame()

        # return response.json({'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}}, default=str)
        return response.json({'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}})

    @web_expt()
    async def get_completed_orders(self, request):
        start_date = request.args.get('start_date')
        completed_orders = self._account.completed_orders
        if completed_orders.shape[0] > 0:
            if start_date is not None:
                completed_orders = completed_orders.loc[completed_orders['updated_time'] >= start_date]
        else:
            completed_orders = pd.DataFrame()
        return response.json(
            {'ret_code': 1, 'return': {'content': {'completed_orders': completed_orders.to_dict('records')}}})

    @web_expt()
    async def web_subscribe_tickers(self, request):
        tickers = eval(request.args.get('tickers'))
        new_tickers = list(set(tickers).difference(self._data.universe))
        self._data.subscribe_tickers(tickers=new_tickers)
        return response.json({'ret_code': 1, 'return': {'content': {'universe': list(self._data.universe),
                                                                    'datatypes': list(self._data.datatypes)}}})

    @web_expt()
    async def unsubscribe_ticker(self, request):
        tickers = eval(request.args.get('tickers'))
        self._data.unsubscribe_tickers(tickers=tickers)

        return response.json({'ret_code': 1, 'return': {'content': {'universe': list(self._data.universe),
                                                                    'datatypes': list(self._data.datatypes)}}})

    @web_expt()
    async def pause(self, request):
        self._running = False
        return response.json({'ret_code': 1, 'return': {'content': {'running': self._running}}})

    @web_expt()
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


class CandlestickStrategy(Algo):
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


class Backtest(Algo):
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
        succeed, failed = self._data.download_ticker_lot_size(tickers=self._data.universe)
        for ticker in succeed:
            self._account.add_new_position(ticker)

        self._logger.debug(f'Loading Date from MySQL DB...')
        backtest_df = pd.DataFrame()
        for tk in self._data.universe:
            for dtype in self._data.datatypes:
                ret_code, df = self._data.download_historical(ticker=tk, datatype=dtype, start_date=start_date,
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
                    self._data.add_cache(datatype=dtype, df=filler, ticker=tk)
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
            self._account.log(overwrite_date=backtest_df.iloc[0]['datetime'].date())
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
                        self._account.log(overwrite_date=last_date)

                self._account.update_prices(datatype=datatype, df=cur_df)
                self._data.add_cache(datatype=datatype, df=cur_df, ticker=ticker)
                trigger_strat, (tgr_dtype, tgr_ticker, tgr_df) = self.determine_trigger(datatype=datatype,
                                                                                        ticker=ticker, df=cur_df)
                if trigger_strat:
                    await self.trigger_strat(datatype=tgr_dtype, ticker=tgr_ticker, df=tgr_df)

            self._logger.debug('Backtesting Completed! Call report() method to see backtesting result!')

        asyncio.run(_backtest())

    def trade(self, ticker, trade_side, order_type, quantity, price):
        lot_size = self.get_lot_size(ticker=ticker)
        risk_passed, msg = self._account.pre_trade_check(ticker=ticker, quantity=quantity, trade_side=trade_side,
                                                         price=price, lot_size=lot_size)
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
        self._account.update_positions(df=order_update_df)
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
        completed_orders = self._account.completed_orders
        orders_df = completed_orders.loc[completed_orders.ticker == ticker].rename(
            columns={'created_time': 'datetime'})
        ticker_df = self._data.cache[datatype][ticker]
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

        PV = self._account.records['PV']
        PV.index = pd.to_datetime(PV.index)
        PV.index.name = 'datetime'
        PV = PV.resample('1D').last().fillna(method='ffill')
        html = f'{self.name}.html'
        qs.reports.html(PV, benchmark, output=html, title=f'{self.name}')
        webbrowser.open(html)


if __name__ == '__main__':
    pass
