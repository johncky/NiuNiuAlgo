# FutuHook
<p>Algorithmic trading system for algo trading on FutuNiuNiu broker (project is not finished)</p>
<hr/>
There are 3 things: FutuHook, Dashboard app, Algo class
<br/>
<img src="https://github.com/johncky/FutuHook/blob/master/docs/System.png">

 
 
<h2>FutuHook</h2>
Functions: 
<ol>
<li>Maintain Connections to Futu OpenD, stream and boradcast real-time data such as 1-min,5-min,15-min candlesticks, orderbook, quotes etc throught ZMQ PUB to connected Algos</li>
<li>Store streamed historical data to MySQL databases</li>
<li>RESTful API for changing subscribed tickers & datatypes, get historical data, download to csv etc</li>
<li>API for placing orders, modifying orders, cancelling orders</li>
</ol>


<h3>How to run?</h3>
<ol>
<li>Install and run FutuOpenD here: <a href='https://www.futunn.com/download/openAPI?lang=en-US'>https://www.futunn.com/download/openAPI?lang=en-US</a></li>
<li>Install and run MySQL database: <a href='https://www.mysql.com/downloads/'>https://www.mysql.com/downloads/</a></li>
<li>Set Up system environment variables for the following:
    <ol>
        <li>SANIC_HOST : host for sanic app (e.g. 0.0.0.0)</li>
        <li>SANIC_PORT: port for sanic app (e.g. 8000)</li>
        <li>FUTU_TRADE_PWD: the trade unlock password for your Futu account</li>
        <li>FUTU_HOST: host of your running Futu OpenD</li>
        <li>FUTU_PORT: port of your running Futu OpenD</li>
        <li>ZMQ_PORT: port for ZMQ</li>
        <li>MYSQL_DB: name of the database that you wish to store your data</li>
        <li>MYSQL_HOST: host of your running MySQL service</li>
        <li>MYSQL_USER: user of your running MySQL service</li>
        <li>MYSQL_PWD: password of your user</li>
    </ol>
</li>
<li>You may create an instance of the FutuHook class and call FutuHook.run(), or simply run the script</li>

```python
    INIT_DATATYPE = ['K_3M', 'K_5M', 'QUOTE']
    INIT_TICKERS = ['HK.00700', 'HK_FUTURE.999010']
    futu_hook = FutuHook()
    # Subscribe to datatypes and tickers
    futu_hook.subscribe(datatypes=INIT_DATATYPE, tickers=INIT_TICKERS)
    futu_hook.run()
```    
<li>Subscribe to new datatypes or tickers, for supported datatypes / tickers, please go to Futu-api documentations:<a href='https://futunnopen.github.io/futu-api-doc/intro/intro.html'>https://futunnopen.github.io/futu-api-doc/intro/intro.html</a></li>
<li>Use POST request to the sanic app to change subscriptions, or subscribe it before you call run() 
<br/><a href='http://0.0.0.0:8000//subscriptions'>http://0.0.0.0:8000//subscriptions</a> with params method='subscribe', datatypes='['K_1M', 'K_3M']', tickers=['HK.00700', 'HK.01299']</li>

```python
    futu_hook.subscribe(datatypes=['K_DAY'], tickers=['HK.800000'])
```
</ol>
<hr/>

<h3>Some API</h3>
    <ol>
        <li>/subscriptions
        <br/>Method: GET
        <br/>Get the current subscriptions
        </li>
        <li>/subscriptions
        <br/>Method: POST
        <br/>Subscribe / Unsubscribe to datatypes and tickers</li>
        <li>/db/fill
        <br/>Method: POST
        <br/>Fill your DataBase with the data of that datatype for a ticker between start_date and end_date (you must subscribe to that datatype and ticker first)</li>
        <li>/download
        <br/>Method: GET
        <br/>download the data to csv file
        <li>/historicals
        <br/>Method: GET
        <br/>retrieve data from either MySQL databases or through Futu OpenD (specify from_exchange=True)  
    </ol>


<h2>Algo</h2>
Functions: 
<ol>
<li>Subscribe to FutuHook to stream stocks data of different types(orders update, candlesticks, tick etc)</li>
<li>Trigger user-defined events on receiving different datatypes (functions such as on_bar, on_qupte, on_order_update)</li>
<li>Cache received data to pickle files</li>
<li>Place orders, modify orders, cancel orders through FutuHook API</li>
<li> RESTful API for retriving strategy information (returns, parameters, positions, pending orders etc)
</ol>
<hr/>

<h3>Example: SMA Crossover</h3>
<p>

```python
class SMACrossover(CandlestickStrategy):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long + 1)
        self.short = short
        self.long = long
        self._test_df = None

    async def on_bar(self, datatype, ticker, df):
        if datatype == 'K_3M':
            df['SMA_short'] = talib.SMA(df['close'], timeperiod=self.short)
            df['SMA_long'] = talib.SMA(df['close'], timeperiod=self.long)

            sma_short_last = df['SMA_short'].iloc[-2]
            sma_short_cur = df['SMA_short'].iloc[-1]

            sma_long_last = df['SMA_long'].iloc[-2]
            sma_long_cur = df['SMA_long'].iloc[-1]

            print('Datetime: {} , Last: {}/{}, Current: {}/{}'.format(df['datetime'].iloc[-1], sma_short_last,
                                                                      sma_long_last, sma_short_cur, sma_long_cur))
            print('\n')
            self._test_df = df

            if (sma_short_last <= sma_long_last) and (sma_short_cur > sma_long_cur) and (self.get_qty(ticker) == 0):
                print(self.buy_limit(ticker=ticker,  quantity=self.get_lot_size(ticker),
                                     price=self.get_price(ticker=ticker)))

            elif (sma_short_last >= sma_long_last) and (sma_short_cur < sma_long_cur) and (self.get_qty(ticker) > 0):
                print(self.sell_limit(ticker=ticker, quantity=self.get_lot_size(ticker),
                                      price=self.get_price(ticker=ticker)))
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
```

</p>

<h3>Running an Algo</h3>

```python
    algo = SMACrossover(short=10, long=20)
    algo.initialize(initial_capital=100000.0, margin=100000.0, mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='SIMULATE',
                    trading_universe=['HK.00700', 'HK_FUTURE.HSImain'], datatypes=['K_3M'])
    algo.run(5000)
```

<h3> Backtesting </h3>
Currently on candlesticks strategies supports backtesting.(FutuNiuNiu only supports historical data of candlsticks data)
To backtest a strategies, inhertie from Backtest class, and run algo.backtest().
Please Note that you must fill your MySQL database with historical data first. You can do so through FutuHook API /db/fill.

```python
class SMACrossover(Backtest):
    def __init__(self, short, long):
        super().__init__(name='SMA Crossover ({}, {})'.format(short, long), bars_no=long+1)
        self.short = short
        self.long = long

    async def on_bar(self, datatype, ticker, df):
        if datatype == 'K_DAY':
            df['SMA_short'] = talib.SMA(df['close'], timeperiod=self.short)
            df['SMA_long'] = talib.SMA(df['close'], timeperiod=self.long)
            df = df.round({'SMA_short': 2, 'SMA_long': 2})
            sma_short_last = df['SMA_short'].iloc[-2]
            sma_short_cur = df['SMA_short'].iloc[-1]

            sma_long_last = df['SMA_long'].iloc[-2]
            sma_long_cur = df['SMA_long'].iloc[-1]


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
    algo = SMACrossover(short=16,long=32)
    algo.initialize(initial_capital=200000.0, margin=200000.0, mq_ip='tcp://127.0.0.1:8001',
                    hook_ip='http://127.0.0.1:8000',
                    hook_name='FUTU', trading_environment='BACKTEST',
                    trading_universe=['HK.00700', 'HK.54544554','HK.00388'], datatypes=['K_DAY'], spread=0)
    algo.backtest(start_date = '2020-04-01', end_date = '2020-05-01')
```

<h3>Some API </h3>
<p>Inside the Algo.py</p>

<h2>Dashboard</h2>
Functions: 
<ol>
<li>A Sanic webapp that retrieve strategies' performance (sharpe, sortino, annulized return etc) and positions through their APIs</li>
<li>Render and serve dashboard webpages (built with React.js) to show visualize performances of strategies, pause and resume strategies, change strategies' parameters</li>
</ol>

<h3>How to Run?</h3>
in Webapp/app.py:

```python
    if __name__ == '__main__':
        app = WebApp()
        app.run(port=8522, hook_ip='http://127.0.0.1:8000')
```

<h3> Interface </h3>
<img src="https://github.com/johncky/FutuAlgo/blob/master/docs/interface.png">
