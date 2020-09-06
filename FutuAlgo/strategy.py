from abc import ABC, abstractmethod


class Strategy(ABC):
    """ Class that handle event triggering """
    def determine_trigger(self, datatype, ticker, df):
        """ logic that determine when to trigger events, return True in first element when trigger """
        return True, (datatype, ticker, df)

    async def trigger_strat(self, datatype, ticker, df):
        if datatype == 'TICKER':
            await self.on_tick(ticker=ticker, df=df)
        elif datatype == 'QUOTE':
            await self.on_quote(ticker=ticker, df=df)
        elif 'K_' in datatype:
            await self.on_bar(datatype=datatype, ticker=ticker, df=df)
        elif datatype == 'ORDER_BOOK':
            await self.on_orderbook(ticker=ticker, df=df)
        else:
            await self.on_other_data(datatype=datatype, ticker=ticker, df=df)

    @abstractmethod
    async def on_other_data(self, datatype, ticker, df):
        pass

    @abstractmethod
    async def on_tick(self, ticker, df):
        pass

    @abstractmethod
    async def on_quote(self, ticker, df):
        pass

    @abstractmethod
    async def on_orderbook(self, ticker, df):
        pass

    @abstractmethod
    async def on_bar(self, datatype, ticker, df):
        pass

    @abstractmethod
    async def on_order_update(self, order_id, df):
        pass
