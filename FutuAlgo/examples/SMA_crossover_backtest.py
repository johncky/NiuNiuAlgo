from FutuAlgo.algo import Backtest
import os
os.environ['PYTHONASYNCIODEBUG'] = '1'


class SMACrossover(Backtest):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long+5)
        self.short = short
        self.long = long
        self._test_df = dict()
        self.rq = dict()

    async def on_bar(self, datatype, ticker, df):
        df['SMA_short'] = df['close'].rolling(self.short).mean()
        df['SMA_long'] = df['close'].rolling(self.long).mean()
        df = df.round({'SMA_short': 2, 'SMA_long': 2})

        sma_short_last = df['SMA_short'].iloc[-2]
        sma_short_cur = df['SMA_short'].iloc[-1]

        sma_long_last = df['SMA_long'].iloc[-2]
        sma_long_cur = df['SMA_long'].iloc[-1]

        if (sma_short_last <= sma_long_last) and (sma_short_cur > sma_long_cur) and (self.get_current_qty(ticker) == 0):
            self.buy_next_open(ticker=ticker, quantity=self.calc_max_buy_qty(ticker), datatype='K_DAY')

        elif (sma_short_last >= sma_long_last) and (sma_short_cur < sma_long_cur) and (self.get_current_qty(ticker) > 0):
            self.sell_next_open(ticker=ticker, quantity=self.get_current_qty(ticker), datatype='K_DAY')

    async def on_order_update(self, order_id, df):
        pass

    async def on_orderbook(self, ticker, df):
        pass

    async def on_other_data(self, datatype, ticker, df):
        pass

    async def on_quote(self, ticker, df):
        pass

    async def on_tick(self, ticker, df):
        pass


if __name__ == '__main__':
    algo = SMACrossover(short=16,long=32)
    algo.initialize(initial_capital=200000.0, mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='BACKTEST',
                    trading_universe=['HK.00700'], datatypes=['K_DAY'], spread=0, txn_cost=0)

    algo.backtest(None, None)