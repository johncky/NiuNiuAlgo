from algo import Algo

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
