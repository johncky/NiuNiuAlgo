class Execution:
    def __init__(self, account, data, trading_environment, logger):
        self._account = account
        self._data = data
        self._logger = logger
        self._trading_environment = trading_environment

    def trade(self, ticker, trade_side, order_type, quantity, price):
        risk_passed, msg = self._account.pre_trade_check(ticker=ticker, quantity=quantity, trade_side=trade_side, price=price)
        if not risk_passed:
            msg = f'Risk check failed:"{order_type} {quantity} qty of {ticker} @ {price}" due to {msg}'
            self._logger.info(msg)
            return 0, msg

        params = {'ticker': ticker, 'trade_side': trade_side, 'order_type': order_type, 'quantity': int(quantity),
                  'price': price, 'trade_environment': self._trading_environment}

        result = self._data.place_order(params=params)
        if result['ret_code'] == 1:
            df = pd.read_json(result['return']['content'], dtype={'order_id': str})
            self._account.update_positions(df=df)
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