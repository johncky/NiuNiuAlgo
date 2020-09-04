from FutuAlgo.Algo import CandlestickStrategy
import datetime
import pandas as pd


class SMACrossover(CandlestickStrategy):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long+1)
        self.short = short
        self.long = long
        self.all_ticks = pd.DataFrame()

    async def on_bar(self, datatype, ticker, df):
        if datatype == 'K_3M':
            self.all_ticks = self.all_ticks.append(df.iloc[-1:])
            self.all_ticks.to_csv('Trade_Recon/tick.csv')
            df['SMA_short'] = df['close'].rolling(self.short).mean()
            df['SMA_long'] = df['close'].rolling(self.long).mean()

            sma_short_last = df['SMA_short'].iloc[-2]
            sma_short_cur = df['SMA_short'].iloc[-1]

            sma_long_last = df['SMA_long'].iloc[-2]
            sma_long_cur = df['SMA_long'].iloc[-1]

            if (sma_short_last <= sma_long_last) and (sma_short_cur > sma_long_cur) and (self.get_qty(ticker) == 0):
                df.to_csv(f'Trade_Recon/BUY_{ticker.split()[1]}_{df["datetime"].iloc[-1]}.csv')
                self.buy_limit(ticker=ticker, quantity=self.get_lot_size(ticker),
                               price=self.get_price(ticker=ticker)+1)


            elif (sma_short_last >= sma_long_last) and (sma_short_cur < sma_long_cur) and (self.get_qty(ticker) > 0):
                df.to_csv(f'Trade_Recon/SELL_{ticker.split()[1]}_{df["datetime"].iloc[-1]}.csv')
                self.sell_limit(ticker=ticker, quantity=self.get_lot_size(ticker),
                                      price=self.get_price(ticker=ticker)-1)
        else:
            pass

    async def on_order_update(self, order_id, df):
        self.logger.info(
            f'{df["order_status"].iloc[-1]} {df["order_type"].iloc[-1]} order to {df["trd_side"].iloc[-1]} {df["dealt_qty"].iloc[-1]}/{df["qty"].iloc[-1]} shares of {df["ticker"].iloc[-1]} @ {df["price"].iloc[-1]}, orderID: {df["order_id"].iloc[-1]}')

    async def on_orderbook(self, ticker, df):
        pass

    async def on_other_data(self, datatype, ticker, df):
        pass

    async def on_quote(self, ticker, df):
        pass

    async def on_tick(self, ticker, df):
        pass


if __name__ == '__main__':
    algo = SMACrossover(short=5, long=10)
    algo.initialize(initial_capital=20000000.0,
                    mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='SIMULATE',
                    trading_universe=['HK.00700', 'HK.09988', 'HK.09999', 'HK.02318', 'HK.02800', 'HK.01211'], datatypes=['K_3M'])
    algo.run(sanic_port=5000, sanic_host='0.0.0.0')
