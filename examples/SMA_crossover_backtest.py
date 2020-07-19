from Algo import Backtest
import datetime
import talib
import os
os.environ['PYTHONASYNCIODEBUG'] = '1'


class SMACrossover(Backtest):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long+1)
        self.short = short
        self.long = long
        self._test_df = None

    async def on_bar(self, datatype, ticker, df):
        if datatype == 'K_1M':
            df['SMA_short'] = talib.SMA(df['close'], timeperiod=self.short)
            df['SMA_long'] = talib.SMA(df['close'], timeperiod=self.long)
            df = df.round({'SMA_short': 2, 'SMA_long': 2})

            sma_short_last = df['SMA_short'].iloc[-2]
            sma_short_cur = df['SMA_short'].iloc[-1]

            sma_long_last = df['SMA_long'].iloc[-2]
            sma_long_cur = df['SMA_long'].iloc[-1]


            self._test_df = df

            # TODO: Wrong price!!!
            if (sma_short_last <= sma_long_last) and (sma_short_cur > sma_long_cur) and (self.get_qty(ticker) == 0):
                self.buy_limit(ticker=ticker,  quantity=self.get_lot_size(ticker),
                                     price=self.get_price(ticker=ticker))

            elif (sma_short_last >= sma_long_last) and (sma_short_cur < sma_long_cur) and (self.get_qty(ticker) > 0):
                self.sell_limit(ticker=ticker, quantity=self.get_lot_size(ticker),
                                      price=self.get_price(ticker=ticker))
        else:
            pass

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
    algo = SMACrossover(short=30,long=60)
    algo.initialize(initial_capital=200000.0, margin=200000.0, mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='BACKTEST',
                    trading_universe=['HK.00700', 'HK.54544554','HK.00388'], datatypes=['K_1M'])
    # algo.run(5000)
    # algo.backtest('2020-04-01', '2020-05-01')