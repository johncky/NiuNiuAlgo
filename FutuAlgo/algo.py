from account import Account
from execution import Execution
from FutuAlgo import logger
from strategy import Strategy
from data import Data
from web import AlgoApp
from tools import *
import pickle
import asyncio
import time


class Algo(Strategy):
    """ Class that ensemble Account, Strategy, Data and AlgoWeb to run algo trading """
    def __init__(self, name: str, log_path='.', benchmark: str = 'HSI'):
        self.name = name
        self.benchmark = benchmark
        self._logger = logger.RootLogger(root_name=self.name, file_path=log_path)
        self._running = False

        self._initialized_date = None
        self._initialized = False

        self._data = None
        self._account = None
        self._execution=None
        self._webapp = None

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
        self._webapp = AlgoApp(algo=self)

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
            async def _run():

                tasks = list()
                web_server = self._webapp.get_coroutine(host=sanic_host, port=sanic_port)
                tasks.append(web_server)
                tasks.append(self.main())
                tasks.append(self.record_daily_performance())
                await asyncio.gather(*tasks)

            loop.create_task(_run())
            loop.run_forever()

    # ------------------------------------------------ [ Trade API ] ------------------------------------------
    def trade(self):
        pass

    def buy_market(self, ticker, quantity):
        return self._execution.buy_market(ticker=ticker, quantity=quantity)

    def sell_market(self, ticker, quantity):
        return self._execution.sell_market(ticker=ticker, quantity=quantity)

    def buy_limit(self, ticker, quantity, price):
        return self._execution.buy_limit(ticker=ticker, quantity=quantity, price=price)

    def sell_limit(self, ticker, quantity, price):
        return self._execution.sell_limit(ticker=ticker, quantity=quantity, price=price)

    # ------------------------------------------------ [ Get Set ] ------------------------------------------
    def get_data(self, datatype, ticker: str, start_date: datetime.datetime = None, n_rows: int = None, sort_drop=True):
        return self._data.get_data(datatype=datatype, ticker=ticker, start_date=start_date, n_rows=n_rows,
                                   sort_drop=sort_drop)

    def get_current_qty(self, ticker):
        return self._account.get_current_qty(ticker=ticker)

    def get_latest_price(self, ticker):
        return self._account.get_latest_price(ticker=ticker)

    def get_lot_size(self, ticker):
        return self._data.get_lot_size(ticker=ticker)

    def calc_max_buy_qty(self, ticker, cash=None, adjust_limit=1.03):
        lot_size = self.get_lot_size(ticker=ticker)
        return self._account.calc_max_buy_qty(ticker=ticker, lot_size=lot_size, cash=cash, adjust_limit=adjust_limit)

    def subscribe_tickers(self, tickers):
        self._data.subscribe_tickers(tickers=tickers)

    def unsubscribe_tickers(self, tickers):
        self._data.unsubscribe_tickers(tickers=tickers)

    def pause(self):
        self._running = False

    def resume(self):
        self._running = True

    @property
    def cash(self):
        return self._account.cash

    @property
    def mv(self):
        return self._account.mv

    @property
    def pv(self):
        return self._account.pv

    @property
    def n_trades(self):
        return self._account.n_trades

    @property
    def init_capital(self):
        return self._account.init_capital

    @property
    def total_txn_cost(self):
        return self._account.total_txn_cost

    @property
    def initialized_date(self):
        return self._initialized_date

    @property
    def running(self):
        return self._running

    @property
    def records(self):
        return self._account.records

    @property
    def pending_orders(self):
        return self._account.pending_orders

    @property
    def completed_orders(self):
        return self._account.completed_orders

    @property
    def positions(self):
        return self._account.positions

    @property
    def universe(self):
        return self._data.unverse

    @property
    def datatypes(self):
        return self._data.datatypes

    @property
    def initialized(self):
        return self._initialized


if __name__ == '__main__':
    pass
