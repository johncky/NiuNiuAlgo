from queue import Queue
import asyncio
from futu import *
from sanic import Sanic
from sanic import response
import zmq
import zmq.asyncio
import pickle
import aiomysql
import numpy as np
import Logger
import collections

d_types = ('K_DAY', 'K_1M', 'K_3M', 'K_5M', 'K_15M', 'QUOTE', 'ORDER_UPDATE')

schemas_sql = {'K': """CREATE TABLE IF NOT EXISTS `{}`.`FUTU_{}` (
  `ticker` VARCHAR(40) NOT NULL,
  `datetime` DATETIME NOT NULL,
  `open` FLOAT NULL,
  `high` FLOAT NULL,
  `low` FLOAT NULL,
  `close` FLOAT NULL,
  `volume` FLOAT NULL,
  `turnover` FLOAT NULL,
  PRIMARY KEY (`ticker`, `datetime`));
""",
               'ORDER_UPDATE': """CREATE TABLE IF NOT EXISTS `{}`.`FUTU_ORDER_UPDATE` (
  `trd_side` VARCHAR(10) NULL,
  `order_type` VARCHAR(20) NULL,
  `order_id` VARCHAR(50) NOT NULL,
  `ticker` VARCHAR(40) NULL,
  `stock_name` VARCHAR(50) NULL,
  `qty` FLOAT NULL,
  `price` FLOAT NULL,
  `create_time` DATETIME NULL,
  `updated_time` DATETIME NULL,
  `dealt_qty` FLOAT NULL,
  `dealt_avg_price` FLOAT NULL,
  `trd_env` VARCHAR(40) NULL,
  `order_status` VARCHAR(40) NULL,
  `trd_market` VARCHAR(40) NULL,
  `last_err_msg` VARCHAR(200) NULL,
  `remark` VARCHAR(200) NULL,
  PRIMARY KEY (`order_id`));""",
               'QUOTE': """CREATE TABLE IF NOT EXISTS `{}`.`FUTU_QUOTE` (
                 `ticker` VARCHAR(40) NOT NULL,
                 `datetime` DATETIME NOT NULL,
                 `quote` FLOAT NULL,
                 `volume` FLOAT NULL,
                 PRIMARY KEY (`ticker`, `datetime`));"""
               }


class FutuKlineHandler(CurKlineHandlerBase):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_recv_rsp(self, rsp_str):
        ret_code, df = super(FutuKlineHandler, self).on_recv_rsp(rsp_str)
        k_type = df.k_type[0].upper()
        if ret_code == RET_OK:
            df = df[['code', 'time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover']].rename(
                columns={'time_key': 'datetime', 'code': 'ticker'})
            df['datetime'] = pd.to_datetime(df['datetime'])
            self.queue.put((f'FUTU.{k_type}.{df["ticker"].iloc[0]}', df))
        return RET_OK, df


class FutuQuoteHandler(StockQuoteHandlerBase):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_recv_rsp(self, rsp_pb):
        ret_code, df = super(FutuQuoteHandler, self).on_recv_rsp(rsp_pb)
        if ret_code == RET_OK:
            df['datetime'] = df['data_date'] + ' ' + df['data_time']
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df[['code', 'datetime', 'last_price', 'volume']].rename(
                columns={'last_price': 'quote', 'code': 'ticker'})
            df['datetime'] = pd.to_datetime(df['datetime'])
            self.queue.put((f'FUTU.QUOTE.{df["ticker"].iloc[0]}', df))
        return RET_OK, df


class FutuOrderUpdateHandler(TradeOrderHandlerBase):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def on_recv_rsp(self, rsp_pb):
        ret, df = super(FutuOrderUpdateHandler, self).on_recv_rsp(rsp_pb)
        if ret == RET_OK:
            df = df.rename(columns={'code': 'ticker'})
            # df['create_time'] = pd.to_datetime(df['create_time'])
            # df['update_time'] = pd.to_datetime(df['update_time'])
            self.queue.put((f'FUTU.ORDER_UPDATE.{df["order_id"].iloc[0]}', df))
            print(df['order_status'])
        return ret, df


class FutuHook():
    def __init__(self, db_autosave_frequency=30 * 60):
        # Get Environment Variables
        self.SANIC_HOST = os.getenv('SANIC_HOST')
        self.SANIC_PORT = os.getenv('SANIC_PORT')
        self.FUTU_TRADE_PWD = os.getenv('FUTU_TRADE_PWD')
        self.FUTU_HOST = os.getenv('FUTU_HOST')
        self.FUTU_PORT = int(os.getenv('FUTU_PORT'))
        self.ZMQ_PORT = os.getenv('ZMQ_PORT')
        self.MYSQL_DB = os.getenv('MYSQL_DB')
        self.MYSQL_HOST = os.getenv('MYSQL_HOST')
        self.MYSQL_USER = os.getenv('MYSQL_USER')
        self.MYSQL_PWD = os.getenv('MYSQL_PWD')

        # Queue
        self.queue = Queue()

        # Futu Context and Trade Context
        SysConfig.set_all_thread_daemon(True)
        self.quote_manager = OpenQuoteContext(host=self.FUTU_HOST, port=self.FUTU_PORT)

        self.trade_manager = dict()
        self.trade_manager['HK'] = OpenHKTradeContext(host=self.FUTU_HOST, port=self.FUTU_PORT)
        self.trade_manager['US'] = OpenUSTradeContext(host=self.FUTU_HOST, port=self.FUTU_PORT)
        self.trade_manager['CN'] = OpenCNTradeContext(host=self.FUTU_HOST, port=self.FUTU_PORT)
        for market in ('HK', 'US', 'CN'):
            self.trade_manager[market].set_handler(
                FutuOrderUpdateHandler(queue=self.queue))
        self.quote_manager.set_handler(FutuQuoteHandler(queue=self.queue))
        self.quote_manager.set_handler(FutuKlineHandler(queue=self.queue))

        # Logger
        self.logger = Logger.RootLogger('FutuHook')
        self.logger.debug('FutuHook trade unlock status {}'.format(self.unlock_trade()))

        # Create a dictionary for storing temporary dfs, which will be saved to MySQL later
        self.tmp_records = collections.defaultdict(lambda: pd.DataFrame())

        self._db_autosave_frequency = db_autosave_frequency
        self._db_last_save_time = time.time()

        # ZMQ Publish and Pair sockets
        self.zmq_context = zmq.asyncio.Context()
        self.mq_socket = self.zmq_context.socket(zmq.PUB)
        self.mq_socket.bind("tcp://127.0.0.1:{}".format(self.ZMQ_PORT))
        self.hello_socket = self.zmq_context.socket(zmq.PAIR)
        self.hello_socket.bind("tcp://127.0.0.1:{}".format(int(self.ZMQ_PORT) + 1))
        self.logger.debug('ZMQ publisher binded @ {}'.format("tcp://127.0.0.1:{}".format(self.ZMQ_PORT)))
        self.logger.debug('ZMQ pair binded @ {}'.format("tcp://127.0.0.1:{}".format(int(self.ZMQ_PORT) + 1)))

        # Sanic App
        self.app = Sanic('FutuHook')

    def unsubscribe(self, datatypes: list, tickers: list, unsubscribe_all=False):
        return self.quote_manager.unsubscribe(tickers, datatypes, unsubscribe_all=unsubscribe_all)

    def subscribe(self, datatypes: list, tickers: list):
        return_content = self.quote_manager.subscribe(code_list=tickers, subtype_list=datatypes, is_first_push=False)
        time.sleep(1)
        while not self.queue.empty():
            self.queue.get()
            self.queue.task_done()
        return return_content

    def unlock_trade(self):
        ret_code_hk, data = self.trade_manager['HK'].unlock_trade(self.FUTU_TRADE_PWD)
        ret_code_us, data = self.trade_manager['US'].unlock_trade(self.FUTU_TRADE_PWD)
        ret_code_cn, data = self.trade_manager['CN'].unlock_trade(self.FUTU_TRADE_PWD)

        if RET_ERROR in (ret_code_hk, ret_code_cn, ret_code_us):
            raise Exception("FutuHook: Trade Unlocked Failed")
        return 'HK:{}   US:{}   CN:{}'.format(ret_code_hk, ret_code_us, ret_code_cn)

    def query_subscriptions(self):
        ret_code, sub = self.quote_manager.query_subscription()
        return ret_code, sub['sub_list']

    async def publish(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.01)
            else:
                while not self.queue.empty():
                    topic, df = self.queue.get()
                    datatype = topic.split('.')[1]
                    self.tmp_records[datatype] = self.tmp_records[datatype].append(df)
                    await self.mq_socket.send_multipart([bytes(topic, 'utf-8'), pickle.dumps(df)])
                    print(topic)
                    self.queue.task_done()

            if (time.time() - self._db_last_save_time) > self._db_autosave_frequency:
                for dtype in d_types:
                    if not self.tmp_records[dtype].empty:
                        df = self.tmp_records[dtype]
                        if ('datetime' in df.columns) and ('ticker' in df.columns):
                            df = df.drop_duplicates(subset=['ticker', 'datetime'], keep='last')

                        self.tmp_records[dtype] = pd.DataFrame()
                        await self.insert_data(datatype=dtype, df=df)

                self._db_last_save_time = time.time()

    async def ping_pong(self):
        while True:
            msg = await self.hello_socket.recv_string()
            if msg == 'Ping':
                await self.hello_socket.send_string('Pong')

    def run(self):
        # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        loop = asyncio.get_event_loop()

        async def _run():
            # Sanic app
            tasks = list()
            self.app_add_route(app=self.app)
            web_server = self.app.create_server(return_asyncio_server=True, host=self.SANIC_HOST, port=self.SANIC_PORT)
            tasks.append(web_server)
            tasks.append(self.publish())
            tasks.append(self.ping_pong())
            await self.db_create_schemas()
            await asyncio.gather(*tasks)

        loop.create_task(_run())
        loop.run_forever()

    # -----------------------------[ MYSQL ] -------------------------------------------
    async def db_create_schemas(self):
        conn = await self.db_get_conn()
        cursor = await conn.cursor()

        for dtype in d_types:
            if 'K_' in dtype:
                try:
                    await cursor.execute(schemas_sql['K'].format(self.MYSQL_DB, dtype))
                    await conn.commit()
                except Exception as e:
                    self.logger.error('SqlManager: failed to create table for {}, reason: {}'.format(dtype, e))

            elif (dtype == 'ORDER_UPDATE') or (dtype == 'QUOTE'):
                try:
                    await cursor.execute(schemas_sql[dtype].format(self.MYSQL_DB))
                    await conn.commit()
                except Exception as e:
                    self.logger.error('SqlManager: failed to create table for {}, reason: {}'.format(dtype, e))

        conn.close()

    async def insert_data(self, datatype, df):

        table = 'FUTU' + '_' + datatype

        if 'K_' in datatype:
            pk_positions = [0, 1]

        elif datatype == 'ORDER_UPDATE':
            pk_positions = [3]

        elif datatype == 'QUOTE':
            pk_positions = [0, 1]

        else:
            return
        self.logger.debug('SqlManager: inserting data {} into Mysql'.format(str(datatype)))
        return await self.insert_df(db=self.MYSQL_DB, table=table, pk_positions=pk_positions, df=df)

    async def insert_df(self, db, table, pk_positions, df):
        conn = await self.db_get_conn()
        cur = await conn.cursor()

        try:
            if df.shape[0] <= (10 ** 4):
                chunks = [df]
            else:
                chunks = np.array_split(df, df.shape[0] / (10 ** 4))

            for idx, df in enumerate(chunks):
                column_name_ok = str()
                column_name = list(df.columns)
                for index, item in enumerate(column_name):
                    item = '`' + item + '`' + ','
                    column_name_ok = column_name_ok + item
                column_name_ok = column_name_ok.rstrip(',')

                df = df.applymap(str)
                # df.replace({'\\': '\\\\'}, regex=True, inplace=True)
                df.replace({'\'': '\\\''}, regex=True, inplace=True)
                temp = df.values.tolist()
                # for row in df.iterrows():
                #     index, data = row
                #     try:
                #         data = data.str.replace('\\', '\\\\')
                #         data = data.str.replace('\'', '\\\'')
                #     except AttributeError:
                #         pass
                #     temp.append(data.tolist())
                value_ok = str()
                for i in temp:
                    for iikey, iivalue in enumerate(i):
                        i[iikey] = '\'' + str(iivalue) + '\''
                    value_ok = value_ok + '(' + ','.join(i) + '),'
                value_ok = value_ok.rstrip(',').replace('\'NULL\'', 'NULL').replace(',\'\'', ',NULL') \
                    .replace('\'-\'', 'NULL').replace('\'nan\'', 'NULL').replace('NaN', '').replace('\'None\'', 'NULL')

                value_position = list()
                for i in range(0, len(df.columns)):
                    if i not in pk_positions:
                        value_position.append('`' + df.columns[i] + '`' + '= VALUES(' + '`' + df.columns[i] + '`' + ')')

                after_duplicate = ','.join(value_position).rstrip(',')
                if len(after_duplicate) <= 0:
                    sql = 'INSERT INTO `%s`.`%s` (%s) values %s;' % \
                          (db, table, column_name_ok, value_ok)
                else:
                    sql = 'INSERT INTO `%s`.`%s` (%s) values %s ON DUPLICATE KEY UPDATE %s;' % \
                          (db, table, column_name_ok, value_ok, after_duplicate)
                await cur.execute(sql)
                await conn.commit()
            return 1, ''
        except Exception as e:
            self.logger.error('SqlManager: exception occur during insert, reason: {}'.format(e))
            return 0, str(e)
        finally:
            conn.close()

    async def db_get_historicals(self, datatype, ticker, start_date: str, end_date: str):
        sql = """SELECT * FROM {}.FUTU_{} where ticker = '{}'""".format(self.MYSQL_DB,
                                                                        datatype, ticker)
        if start_date or end_date:
            sql += """ and """
            if start_date and end_date:
                sql += """ datetime >= '{}' and datetime <= '{}'""".format(start_date, end_date)
            else:
                if start_date is not None:
                    sql += "datetime >= '{}'".format(start_date)
                else:
                    sql += "datetime <= '{}'".format(end_date)
        conn = await self.db_get_conn()
        cur = await conn.cursor()
        await cur.execute(sql)
        df = pd.DataFrame.from_records(list(await cur.fetchall()))
        if df.shape[0] != 0:
            column_names = [i[0] for i in cur.description]
            df.columns = column_names
        conn.close()
        return df

    async def db_get_conn(self):
        try:
            return await aiomysql.connect(host=str(self.MYSQL_HOST),
                                          user=str(self.MYSQL_USER),
                                          password=str(self.MYSQL_PWD),
                                          db=str(self.MYSQL_DB),
                                          charset='utf8mb4')
        except Exception as e:
            self.logger.error('SqlManager: failed to get conn, reason: {}'.format(e))

    # -----------------------------[ Sanic ] -------------------------------------------

    # GET: None
    async def get_subscriptions(self, request):
        ret_code, sub = self.quote_manager.query_subscription()
        data = {'ret_code': 1 if ret_code != RET_ERROR else 0, 'return': {'content': sub}}
        return response.json(data)

    # POST: method, datatypes, tickers
    async def set_subscriptions(self, request):
        method = request.form.get('method').upper()
        dtypes = eval(request.form.get('datatypes'))
        tickers = eval(request.form.get('tickers'))
        if method == 'SUBSCRIBE':
            ret_code, df = self.subscribe(tickers=tickers, datatypes=dtypes)
            if ret_code == RET_OK:
                ret_code, sub = self.quote_manager.query_subscription()
                data = {'ret_code': 1, 'return': {'content': sub}}
            else:
                data = {'ret_code': 1, 'return': {'content': df}}
            return response.json(data)
        elif method == 'UNSUBSCRIBE':
            ret_code, df = self.unsubscribe(tickers=tickers, datatypes=dtypes)
            if ret_code == RET_OK:
                ret_code, sub = self.quote_manager.query_subscription()
                data = {'ret_code': 1, 'return': {'content': sub}}
            else:
                data = {'ret_code': 1, 'return': {'content': df}}
            return response.json(data)

    # GET: datatype, ticker, start_date, end_date, from_exchange
    async def download_historicals(self, request):
        datatype = request.args.get('datatype')
        assert datatype in d_types, 'Invalid data type'
        ticker = request.args.get('ticker')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        end_date = datetime.today().strftime('%Y-%m-%d') if end_date is None else end_date

        df = await self.db_get_historicals(datatype=datatype, ticker=ticker, start_date=start_date,
                                           end_date=end_date)
        if self.tmp_records[datatype].shape[0] != 0:
            df = df.append(
                self.tmp_records[datatype].loc[self.tmp_records[datatype]['ticker'] == ticker])
        if df.shape[0] != 0:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.drop_duplicates(
                subset=['datetime', 'ticker'], keep='last').sort_values(by='datetime', axis=0)

        output_locations = './outputs/{}_{}.csv'.format(datatype, ticker)
        df.to_csv(output_locations)
        return response.json({'ret_code': 1, 'return': {'content': output_locations}})

    # GET: datatype, ticker, start_date, end_date, from_exchange
    async def get_historicals(self, request):
        datatype = request.args.get('datatype')
        assert datatype in d_types, 'Invalid data type'
        from_exchange = True if request.args.get('from_exchange').upper() == 'TRUE' else False
        ticker = request.args.get('ticker')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        end_date = datetime.today().strftime('%Y-%m-%d') if end_date is None else end_date

        if from_exchange:
            result = self.quote_manager.request_history_kline(code=ticker, ktype=datatype,
                                                              start=start_date,
                                                              end=end_date, max_count=None)
            ret_code, df = result[0], result[1]
            if ret_code == -1:
                return response.json({'ret_code': 0, 'return': {'content': df}})

            df = df[['code', 'time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover']].rename(
                columns={'time_key': 'datetime', 'code': 'ticker'})
            df['datetime'] = pd.to_datetime(df['datetime'])
            return response.json({'ret_code': 1, 'return': {'content': df.to_json()}})

        else:
            df = await self.db_get_historicals(datatype=datatype, ticker=ticker, start_date=start_date,
                                               end_date=end_date)
            if self.tmp_records[datatype].shape[0] != 0:
                df = df.append(
                    self.tmp_records[datatype].loc[self.tmp_records[datatype]['ticker'] == ticker])
            if df.shape[0] != 0:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.drop_duplicates(
                    subset=['datetime', 'ticker'], keep='last').sort_values(by='datetime', axis=0)
            df.reset_index(inplace=True, drop=True)
            return response.json({'ret_code': 1, 'return': {'content': df.to_json()}})

    # GET: datatype
    async def db_get_last_update_time(self, request):
        datatype = request.args.get('datatype').upper()
        assert datatype in d_types, 'Invalid data type'
        sql = """ Select ticker, max(datetime) from {}.FUTU_{} group by ticker""".format(self.MYSQL_DB, datatype)
        conn = await self.db_get_conn()
        cur = await conn.cursor()
        await cur.execute(sql)
        df = pd.DataFrame.from_records(list(await cur.fetchall()))
        if df.shape[0] > 0:
            column_names = [i[0] for i in cur.description]
            df.columns = column_names
        conn.close()
        if df.shape[0] == 0:
            return response.json({'ret_code': 1, 'return': {'content': 'DB is empty'}})
        else:
            df['max(datetime)'] = [x.strftime("%Y-%m-%d %H:%M:%S") for x in df['max(datetime)']]
            df = df.set_index('ticker')['max(datetime)'].to_dict()
            return response.json({'ret_code': 1, 'return': {'content': df}})

    # POST: ticker, trade_side, order_type, trade_environment, quantity, price
    async def place_order(self, request):
        trade_side = str(request.form.get('trade_side')).upper()
        ticker = request.form.get('ticker')
        market = ticker.split('.')[0]
        assert market in ('HK', 'US', 'CN'), "Invalid market '{}'".format(market)
        order_type = str(request.form.get('order_type')).upper()
        assert order_type in ('ABSOLUTE_LIMIT', 'MARKET', 'NORMAL'), "Invalid order type '{}'".format(order_type)
        assert trade_side in ('BUY', 'SELL', 'BUY_BACK', 'SELL_SHORT'), "Invalid trade side '{}'".format(trade_side)
        trade_environment = str(request.form.get('trade_environment')).upper()
        assert trade_environment in ('REAL', 'SIMULATE',), "Invalid trade environment '{}'".format(trade_environment)
        assert ticker is not None, "ticker cannot be empty"
        ticker = str(ticker).upper()
        quantity = request.form.get('quantity')
        assert quantity is not None, 'quantity cannot be empty'
        quantity = int(quantity)
        if order_type == 'MARKET':
            ret_code, df = self.trade_manager[market].place_order(price=0.0, qty=abs(quantity),
                                                                  code=ticker, order_type=order_type,
                                                                  trd_env=trade_environment, trd_side=trade_side)
            price = 'MARKET'
        else:
            price = request.form.get('price')
            assert price is not None, 'price cannot be empty'
            price = float(price)
            adjust_limit = request.form.get('adjust_limit')
            adjust_limit = 0.0 if adjust_limit is None else float(adjust_limit)
            ret_code, df = self.trade_manager[market].place_order(price=price, qty=abs(quantity),
                                                                  code=ticker, order_type=order_type,
                                                                  trd_env=trade_environment, trd_side=trade_side,
                                                                  adjust_limit=adjust_limit)
        if ret_code == RET_OK:
            order_id = df['order_id'].iloc[0]
            self.logger.info(
                'ORDER: Placed order to buy {} shares of {} @ {}, order_id: {}'.format(quantity, ticker, price,
                                                                                       order_id))
            return response.json({'ret_code': 1, 'return': {'content': df.to_json(), 'order_id': order_id}})
        else:
            self.logger.info(
                'ORDER: Failed to buy {} shares of {} @ {}, reason: {}'.format(quantity, ticker, price, df))
            return response.json({'ret_code': 0, 'return': {'content': df}})

    # POST: ticker, order_id, trade_env, action, quantity, price
    async def modify_order(self, request):
        order_id = request.form.get('order_id')
        assert order_id is not None, 'order_id cannot be empty'

        action = str(request.form.get('action')).upper()
        assert action in ('NORMAL', 'CANCEL', 'DISABLE', 'ENABLE', 'DELETE'), "Invalid action '{}'".format(action)

        trade_environment = str(request.form.get('trade_environment')).upper()
        assert trade_environment in ('REAL', 'SIMULATE',), "Invalid trade environment '{}'".format(
            trade_environment)

        ticker = request.form.get('ticker')
        market = ticker.split('.')[0]

        if action == 'NORMAL':
            quantity = request.form.get('quantity')
            assert quantity is not None, 'quantity cannot be empty'
            quantity = float(quantity)

            price = request.form.get('price')
            assert price is not None, 'price cannot be empty'
            price = float(price)

            adjust_limit = request.form.get('adjust_limit')
            adjust_limit = 0.0 if adjust_limit is None else float(adjust_limit)

        else:
            quantity = 0.0
            price = 0.0
            adjust_limit = 0.0

        ret_code, df = self.trade_manager[market].modify_order(order_id=order_id,
                                                               modify_order_op=action,
                                                               price=price, qty=abs(quantity),
                                                               trd_env=trade_environment,
                                                               adjust_limit=adjust_limit)

        if ret_code == RET_OK:
            return response.json({'ret_code': 1, 'return': {'content': df.to_json(), 'order_id': order_id}})
        else:
            return response.json({'ret_code': 0, 'return': {'content': df}})

    # POST: ticker, order_id, trade_env
    async def cancel_order(self, request):
        request.form['action'] = ['CANCEL']
        return await self.modify_order(request)

    # POST: ticker, datatype, start_date
    async def db_fill_data(self, request):
        ticker = request.form.get('ticker').upper()
        assert ticker is not None, "ticker cannot be empty"
        datatype = request.form.get('datatype').upper()
        assert datatype in d_types, "Invalid data type"
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        end_date = datetime.today().strftime('%Y-%m-%d') if end_date is None else end_date

        ret = self.quote_manager.request_history_kline(code=ticker, ktype=datatype,
                                                       start=start_date,
                                                       end=end_date, max_count=None)
        ret_code = ret[0]
        df = ret[1]
        if ret_code == -1:
            raise Exception(df)
        else:
            df = df[['code', 'time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover']].rename(
                columns={'time_key': 'datetime', 'code': 'ticker'})
            ret_code, msg = await self.insert_data(datatype=datatype, df=df)
            if ret_code:
                return response.json({'ret_code': 1, 'return': {'content': 'Successful'}})
            else:
                return response.json({'ret_code': 0, 'return': {'content': 'DB Insertion failed: {}'.format(msg)}})

    # GET: tickers
    async def get_lot_size(self, request):
        tickers = request.args.get('tickers')
        assert tickers is not None, 'tickers cannot be empty'
        if ('[' in tickers) and (']' in tickers):
            tickers = eval(tickers)
        else:
            raise Exception('tickers should be a list')
        ret_code, data = self.quote_manager.get_stock_basicinfo(market=tickers[0].upper().split('.')[0],
                                                                code_list=tickers)
        if ret_code == 0:
            df = data[['code', 'lot_size']].rename(columns={'code': 'tickers'}).set_index('tickers', drop=True)
            return response.json({'ret_code': 1, 'return': {'content': df.to_json()}})
        else:
            raise Exception(data)

    def app_add_route(self, app):
        app.add_route(self.get_subscriptions, '/subscriptions', methods=['GET'])
        app.add_route(self.set_subscriptions, '/subscriptions', methods=['POST'])
        app.add_route(self.get_historicals, '/historicals', methods=['GET'])
        app.add_route(self.download_historicals, '/download', methods=['GET'])
        app.add_route(self.db_get_last_update_time, '/db/last_update', methods=['GET'])
        app.add_route(self.place_order, '/order/place', methods=['POST'])
        app.add_route(self.modify_order, '/order/modify', methods=['POST'])
        app.add_route(self.cancel_order, '/order/cancel', methods=['POST'])
        app.add_route(self.db_fill_data, '/db/fill', methods=['POST'])
        app.add_route(self.get_lot_size, '/order/lot_size', methods=['GET'])


if __name__ == '__main__':
    # Start FutuHook
    INIT_DATATYPE = ['K_3M', 'K_5M', 'QUOTE']
    INIT_TICKERS = ['HK.00700', 'HK.09988', 'HK.09999', 'HK.02318', 'HK.02800']
    futu_hook = FutuHook()
    futu_hook.subscribe(datatypes=INIT_DATATYPE, tickers=INIT_TICKERS)
    futu_hook.run()


