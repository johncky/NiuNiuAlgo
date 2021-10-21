# FutuHook
algo trading on FutuNiuNiu broker </p>

### Dashboard Interface
<img src="https://github.com/johncky/FutuAlgo/blob/master/docs/interface.png">

## Dashboard
Functions: 
1. Retrieves strategies' performances and positions
2. Serve dashboard webpages 
3. Pause, Resume, Tune Algos

### Run
```python
    import FutuAlgo
    
    if __name__ == '__main__':
        app = FutuAlgo.WebApp()
        app.run(port=8522, hook_ip='http://127.0.0.1:8000')
```

## FutuHook
1. Maintain Connection to Futu OpenD, broadcast data thorugh ZMQ
2. Save data to MySQL db</li>
3. Data downloads, subscription changes</li>
4. Place, modify, cancel orders</li>

### How to run?
1. Install and run FutuOpenD: <a href='https://www.futunn.com/download/openAPI?lang=en-US'>https://www.futunn.com/download/openAPI?lang=en-US</a></li>
2. Install and run MySQL database: <a href='https://www.mysql.com/downloads/'>https://www.mysql.com/downloads/</a></li>
3. Set environment variables:
  - SANIC_HOST : host for sanic app (e.g. 0.0.0.0)</li>
  - SANIC_PORT: port for sanic app (e.g. 8000)</li>
  - FUTU_TRADE_PWD: trade unlock password for Futu </li>
  - FUTU_HOST: host for Futu OpenD</li>
  - FUTU_PORT: port for Futu OpenD</li>
  - ZMQ_PORT: port for ZMQ</li>
  - MYSQL_DB: name of the db</li>
  - MYSQL_HOST: host for MySQL</li>
  - MYSQL_USER: user for MySQL</li>
  - MYSQL_PWD: password for MySQLr</li>

```python
    import FutuAlgo

    INIT_DATATYPE = ['K_3M', 'K_5M', 'QUOTE']
    INIT_TICKERS = ['HK.00700', 'HK_FUTURE.999010']
    futu_hook = FutuAlgo.FutuHook()
    futu_hook.subscribe(datatypes=INIT_DATATYPE, tickers=INIT_TICKERS)
    futu_hook.run(fill_db=True)
```    

## Algo
Functions: 
1. Listen to FutuHook and receive price updates
2. Trigger events on receiving data 
5. Retrieving strategy infos through Sanic(returns, positions, pending orders etc)

### Example: SMA Crossover

```python
import FutuAlgo

class SMACrossover(FutuAlgo.CandlestickStrategy):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long + 1)
        self.short = short
        self.long = long

    async def on_bar(self, datatype, ticker, df):
        df['SMA_short'] = talib.SMA(df['close'], timeperiod=self.short)
        df['SMA_long'] = talib.SMA(df['close'], timeperiod=self.long)

        sma_short_last = df['SMA_short'].iloc[-2]
        sma_short_cur = df['SMA_short'].iloc[-1]

        sma_long_last = df['SMA_long'].iloc[-2]
        sma_long_cur = df['SMA_long'].iloc[-1]

        if (sma_short_last <= sma_long_last) and (sma_short_cur > sma_long_cur) and (self.get_qty(ticker) == 0):
            self.buy_limit(ticker=ticker, quantity=self.cal_max_buy_qty(ticker),
                           price=self.get_price(ticker=ticker))

        elif (sma_short_last >= sma_long_last) and (sma_short_cur < sma_long_cur) and (self.get_qty(ticker) > 0):
            self.sell_limit(ticker=ticker, quantity=self.get_qty(ticker),
                                  price=self.get_price(ticker=ticker))

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
```

### Run an Algo
```python
    algo = SMACrossover(short=10, long=20)
    algo.initialize(initial_capital=100000.0, margin=100000.0, mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='SIMULATE',
                    trading_universe=['HK.00700', 'HK.01299'], datatypes=['K_3M'])
    algo.run(5000)
```

## Backtesting 
```python
import FutuAlgo

class SMACrossover(FutuAlgo.Backtest):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long+1)
        self.short = short
        self.long = long

    async def on_bar(self, datatype, ticker, df):
        df['SMA_short'] = talib.SMA(df['close'], timeperiod=self.short)
        df['SMA_long'] = talib.SMA(df['close'], timeperiod=self.long)
        sma_short_last = df['SMA_short'].iloc[-2]
        sma_short_cur = df['SMA_short'].iloc[-1]

        sma_long_last = df['SMA_long'].iloc[-2]
        sma_long_cur = df['SMA_long'].iloc[-1]


        if (sma_short_last <= sma_long_last) and (sma_short_cur > sma_long_cur) and (self.get_qty(ticker) == 0):
            self.buy_limit(ticker=ticker, quantity=self.cal_max_buy_qty(ticker),
                           price=self.get_price(ticker=ticker))

        elif (sma_short_last >= sma_long_last) and (sma_short_cur < sma_long_cur) and (self.get_qty(ticker) > 0):
            self.sell_limit(ticker=ticker, quantity=self.get_qty(ticker),
                                  price=self.get_price(ticker=ticker))

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
    algo.initialize(initial_capital=200000.0, margin=200000.0, mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='BACKTEST',
                    trading_universe=['HK.00700', 'HK.54544554','HK.00388'], datatypes=['K_DAY'], spread=0)
    algo.backtest(start_date = '2020-04-01', end_date = '2020-05-01')
```

### Backtesting report 
```python
    # Use tencent 0700 as benchmark. This will open a webbrowser showing the full report.
    algo.report(benchmark='0700.HK')
```
