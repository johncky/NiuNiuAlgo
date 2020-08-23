# FutuHook
Algorithmic trading system for algo trading on FutuNiuNiu broker (project is not finished)</p>
<a href="https://github.com/johncky/FutuAlgo/blob/master/docs/System.png">How it works</a>

## FutuHook
1. Maintain Connection to Futu OpenD, stream and broadcast real-time data 
2. Store streamed data to MySQL databases</li>
3. API for changing subscribed tickers & data types, get historical data and more</li>
4. API for placing orders, modifying orders, cancelling orders</li>

### How to run?
1. Install and run FutuOpenD: <a href='https://www.futunn.com/download/openAPI?lang=en-US'>https://www.futunn.com/download/openAPI?lang=en-US</a></li>
2. Install and run MySQL database: <a href='https://www.mysql.com/downloads/'>https://www.mysql.com/downloads/</a></li>
3. Set Up system environment variables for the following:
  - SANIC_HOST : host for sanic app (e.g. 0.0.0.0)</li>
  - SANIC_PORT: port for sanic app (e.g. 8000)</li>
  - FUTU_TRADE_PWD: the trade unlock password for your Futu account</li>
  - FUTU_HOST: host of your running Futu OpenD</li>
  - FUTU_PORT: port of your running Futu OpenD</li>
  - ZMQ_PORT: port for ZMQ</li>
  - MYSQL_DB: name of the database that you wish to store your data</li>
  - MYSQL_HOST: host of your running MySQL service</li>
  - MYSQL_USER: user of your running MySQL service</li>
  - MYSQL_PWD: password of your user</li>

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
1. Subscribe to FutuHook and receive price updates
2. Trigger events on receiving different data types
4. Place orders, modify orders, cancel orders
5. APIs for retrieving strategy infos(returns, params, positions, pending orders ...)

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

### Backtesting - research stage quick back-test</h3>
Quick back-test module for research purpose: [QuickBacktest](http://www.github.com/johncky/QuickBacktest)

```python
    from QuickBacktest.quickBacktest import Strategy

    class SMA(Strategy):
        def init(self):
            self.data['sma16'] = self.data['adjclose'].rolling(16).mean()
            self.data['sma32'] = self.data['adjclose'].rolling(32).mean()
            self.data['dif'] = self.data['sma16'] - self.data['sma32']
            self.data['pre_dif'] = self.data['dif'].shift(1)

        def signal(self):
            if self.dif > 0 and self.pre_dif <= 0:
                self.long()

            elif self.dif < 0 and self.pre_dif >= 0:
                self.exit_long()
            else:
                pass

    tickers = ('FB', 'AMZN', 'AAPL', 'GOOG', 'NFLX', 'MDB', 'NET', 'TEAM', 'CRM')
    sma = SMA()
    result = sma.backtest(tickers=tickers,
                           capital=1000000,
                           start_date="2015-01-01",
                           end_date="2020-07-31",
                           buy_at_open=True,
                           bid_ask_spread=0.0,
                           fee_mode='FIXED:0',
                           data_params={'interval': '1d', 'range': '10y'})
                      
```

### Backtesting report

```python
    result.portfolio_report(benchmark="^IXIC", allocations=[0.25,0.25,0.25,0.25])
    result.ticker_report('FB', benchmark='^IXIC')
```

#### Plot entry and exit points</h4>
```python
    result.ticker_plot('FB')
```

### Backtesting - Using Algo Class
Use Algo class to back-test Strategies. This is useful when you have strategies that uses multiple 
data types and data source.

You need to fill your MySQL database with historical data first, as it retrieve data from your MySQL
database through FutuHook. (This is to avoid wasting download quota of your Futu account)

You can do so by using FutuHook api: /db/fill.

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
Generated by <a href="https://github.com/ranaroussi/quantstats">Quantstats</a>. 

```python
    # Use tencent 0700 as benchmark. This will open a webbrowser showing the full report.
    algo.report(benchmark='0700.HK')
```

### Plot entry and exit points
```python
    # Plot exit-entry point of this ticker
    algo.plot_ticker_trades('K_DAY', 'HK.00700')
```
<img src="https://github.com/johncky/FutuAlgo/blob/master/docs/exit_entry_plot.png">


## Dashboard (in development)
Functions: 
1. Sanic webapp that retrieves strategies' performances and positions
2. Render and serve dashboard webpages 
3. Pause, Resume, Tune your Algos

### How to Run?
```python
    import FutuAlgo
    
    if __name__ == '__main__':
        app = FutuAlgo.WebApp()
        app.run(port=8522, hook_ip='http://127.0.0.1:8000')
```

### Interface
<img src="https://github.com/johncky/FutuAlgo/blob/master/docs/interface.png">

## Notes
1. Cash is deducted after order is filled. If you place a new order before previous one is filled, you might end up with negative cash balance.</li>
