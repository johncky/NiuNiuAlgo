import requests
import itertools
import pandas as pd
import zmq
import zmq.asyncio
import collections
from config import *
from tools import *


class Data:
    """ Class that handle data connection, subscription, download and cache """

    def __init__(self, mq_ip, logger, hook_ip, trading_universe, datatypes: list, cache_rows, test_mq_con,
                 hook_name, prefill_period, add_pos_func):
        self._logger = logger

        self._mq_ip = mq_ip
        self._hook_ip = hook_ip
        self._topics = list()
        self._hook_name = hook_name
        self._zmq_context = None
        self._mq_socket = None

        if test_mq_con:
            self.test_connection()
        self.test_hook_conn()

        self._cache = collections.defaultdict(lambda: collections.defaultdict(lambda: pd.DataFrame()))
        self._cache_rows = cache_rows
        self._prefill_period = prefill_period

        valid_datatypes = list(set(datatypes).intersection(supported_dtypes))
        self._datatypes = valid_datatypes
        self._ticker_lot_size = dict()
        self._trading_universe = trading_universe
        self._failed_tickers = list()
        self._add_pos_func = add_pos_func

    @try_expt(msg='ZMQ Connection failed...', expt=zmq.error.Again)
    def test_connection(self):
        # Test Connection with ZMQ
        test_context = zmq.Context()
        test_socket = test_context.socket(zmq.PAIR)
        test_socket.setsockopt(zmq.LINGER, 0)
        test_socket.setsockopt(zmq.SNDTIMEO, 2000)
        test_socket.setsockopt(zmq.RCVTIMEO, 2000)

        hello_mq_ip = self._mq_ip.split(':')
        hello_mq_ip = ':'.join([hello_mq_ip[0], hello_mq_ip[1], str(int(hello_mq_ip[2]) + 1)])
        test_socket.connect(hello_mq_ip)
        test_socket.send_string('Ping')
        msg = test_socket.recv_string()
        if msg != 'Pong':
            raise Exception()
        self._logger.debug(f'Test Connection with ZMQ {self._mq_ip} is Successful!')
        test_context.destroy()

    @try_expt(msg='FutuHook Connection failed...', expt=requests.ConnectionError)
    def test_hook_conn(self):
        requests.get(self._hook_ip + '/subscriptions').json()
        self._logger.debug(f'Test Connection with FutuHook IP f{self._hook_ip} is Successful!')

    def start_sub(self):
        self._zmq_context = zmq.asyncio.Context()
        self._mq_socket = self._zmq_context.socket(zmq.SUB)
        self._mq_socket.connect(self._mq_ip)
        self.subscribe_tickers(tickers=self._trading_universe)

    # ------------------------------------------------ [ Data ] ------------------------------------------
    def subscribe_tickers(self, tickers):
        if len(tickers) > 0:
            tickers = pd.unique(tickers).tolist()
            self._trading_universe = list(set(self._trading_universe).union(tickers))
            succeed, failed = self.download_ticker_lot_size(tickers=tickers)

            self.download_all_data(tickers=succeed, start_date=period_to_start_date(self._prefill_period))
            new_topics = self.add_zmq_topics(tickers=succeed)

            for ticker in succeed:
                self._add_pos_func(ticker)

            for topic in new_topics:
                self._mq_socket.subscribe(topic)
                self._logger.debug(f'ZMQ subscribed to {topic}')

            for failed_ticker in failed:
                self._logger.debug(f'Failed to subscribe {failed_ticker}')

    def unsubscribe_tickers(self, tickers):
        tickers = list(set(tickers).intersection(self._trading_universe))
        new_topics = self.add_zmq_topics(tickers=tickers)
        for topic in new_topics:
            self._mq_socket.unsubscribe(topic)
        self._topics = list(set(self._topics).difference(new_topics))
        self._trading_universe = list(set(self._trading_universe).difference(tickers))

    def add_cache(self, datatype, ticker, df):
        self._cache[datatype][ticker] = self._cache[datatype][ticker].append(df).drop_duplicates(
            subset=['datetime', 'ticker'], keep='last')
        self._cache[datatype][ticker] = self._cache[datatype][ticker].iloc[-self._cache_rows:]

    def get_data(self, datatype, ticker: str, start_date: datetime.datetime = None, n_rows: int = None, sort_drop=True):
        df = self._cache[datatype][ticker]
        if start_date:
            df = df.loc[df['datetime'] >= start_date]
        if n_rows:
            df = df.iloc[-n_rows:]
        if sort_drop:
            df = df.drop_duplicates(['datetime', 'ticker'], keep='last').sort_values(['datetime'])
        return df

    def download_historical(self, ticker, datatype, start_date=None, end_date=None, from_exchange=False):
        params = {'ticker': ticker, 'datatype': datatype, 'start_date': start_date, 'end_date': end_date,
                  'from_exchange': from_exchange}
        result = requests.get(self._hook_ip + '/historicals', params=params).json()
        if result['ret_code'] == 1:
            df = pd.read_json(result['return']['content'])
            return 1, df
        else:
            return 0, result['return']['content']

    def download_ticker_data(self, ticker, datatype, start_date, from_exchange=False):
        ret_code, df = self.download_historical(ticker=ticker, datatype=datatype, start_date=start_date,
                                                from_exchange=from_exchange)

        if ret_code == 1:
            self.add_cache(datatype=datatype, ticker=ticker, df=df)
        else:
            raise Exception(f'Failed to download historical data from Hook due to {df}')

    def download_all_data(self, start_date, tickers=None):

        tickers = self._trading_universe if tickers is None else tickers
        for dtype in self._datatypes:
            for ticker in tickers:
                self.download_ticker_data(ticker=ticker, datatype=dtype, start_date=start_date)

    def download_ticker_lot_size(self, tickers):
        params = {'tickers': str(tickers)}

        result = requests.get(self._hook_ip + '/order/lot_size', params=params).json()
        if result['ret_code'] == 1:
            lot_size_df = pd.read_json(result['return']['content'])
            failed = list(lot_size_df.loc[lot_size_df['lot_size'] == 0].index)
            succeed = list(lot_size_df.loc[lot_size_df['lot_size'] > 0].index)
            for ticker in succeed:
                self._ticker_lot_size[ticker] = lot_size_df.loc[ticker]['lot_size']
            self._trading_universe = list(set(self._trading_universe).difference(failed))
            self._failed_tickers = list(set(self._failed_tickers).union(failed))
            return succeed, failed
        else:
            raise Exception(f'Failed to request lot size due to {result["return"]["content"]}')

    def add_zmq_topics(self, tickers):
        tmp = [f'{self._hook_name}.{x[0]}.{x[1]}' for x in itertools.product(self._datatypes, tickers)]
        self._topics = list(set(self._topics).union(tmp))
        return tmp

    def place_order(self, params):
        trade_url = self._hook_ip + '/order/place'
        result = requests.post(trade_url, data=params).json()
        if result['ret_code'] == 1:
            order_id = result['return']['order_id']
            self._mq_socket.subscribe(f'{self._hook_name}.ORDER_UPDATE.{order_id}')
        return result

    def get_lot_size(self, ticker):
        return self._ticker_lot_size[ticker]

    async def receive_data(self):
        return await self._mq_socket.recv_multipart()

    @property
    def datatypes(self):
        return self._datatypes.copy()

    @property
    def universe(self):
        return self._trading_universe.copy()

    @property
    def cache(self):
        return self._cache.copy()