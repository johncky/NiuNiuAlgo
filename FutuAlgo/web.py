from tools import web_expt
from sanic import response, Sanic
import datetime
import pandas as pd

class AlgoApp:
    def __init__(self, algo):
        self._algo = algo
        self._ip = None
        self._sanic = None
        self._sanic_host = None
        self._sanic_port = None

    def get_coroutine(self, host, port):
        self._sanic = Sanic(self._algo.name)
        self._sanic_host = host
        self._sanic_port = port
        self._ip = 'http://' + self._sanic_host + ':' + str(self._sanic_port)
        self.app_add_route(app=self._sanic)
        return self._sanic.create_server(return_asyncio_server=True, host=self._sanic_host, port=self._sanic_port)

    @web_expt()
    async def get_attributes(self, request):
        return_attributes = dict()
        for name, value in self._algo.__dict__.items():
            if (type(value) in (list, str, float, int)) and (name[0] != '_'):
                return_attributes[name] = value
        return response.json({'ret_code': 1, 'return': {'content': return_attributes}})

    @web_expt()
    async def get_summary(self, request):
        days_since_deployment = max(int((datetime.datetime.today() - self._algo.initialized_date).days), 1)

        return response.json({'ret_code': 1, 'return': {'content': {'name': self._algo.name,
                                                                    'benchmark': self._algo.benchmark,
                                                                    'status': 'Running' if self._algo.running else 'Paused',
                                                                    'initial_capital': self._algo.init_capital,
                                                                    'ip': self._ip,
                                                                    'pv': self._algo.mv + self._algo.cash,
                                                                    'cash': self._algo.cash,
                                                                    'n_trades': self._algo.n_trades,
                                                                    'txn_cost_total': self._algo.total_txn_cost,
                                                                    'initialized_date': self._algo.initialized_date.strftime(
                                                                        '%Y-%m-%d'),
                                                                    'days_since_deployment': days_since_deployment}}})

    @web_expt()
    async def get_records(self, request):
        start_date = request.args.get('start_date')
        rows = request.args.get('rows')

        record = self._algo.records
        if rows is not None:
            record = record.iloc[-int(rows):]

        if start_date is not None:
            record = record.loc[record.index >= start_date]

        time_series = dict()
        for ts in ('PV', 'EV', 'Cash'):
            ts_series = record[ts].reset_index()
            ts_series.columns = ['x', 'y']
            ts_series = ts_series.to_dict('records')
            time_series[ts] = ts_series

        return response.json({'ret_code': 1, 'return': {'content': time_series}})

    @web_expt()
    async def get_positions(self, request):
        positions = self._algo.positions.reset_index()
        positions.columns = ['ticker', 'price', 'quantity', 'market_value']
        positions = positions.loc[abs(positions['market_value']) > 0]
        return response.json({'ret_code': 1, 'return': {'content': {'positions': positions.to_dict('records')}}})

    @web_expt()
    async def get_pending_orders(self, request):
        start_date = request.args.get('start_date')
        pending_orders = self._algo.pending_orders
        if len(pending_orders) > 0:
            pending_orders = pd.concat(pending_orders.values(), axis=1)

            if start_date is not None:
                pending_orders = pending_orders.loc[pending_orders['updated_time'] >= start_date]
        else:
            pending_orders = pd.DataFrame()

        # return response.json({'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}}, default=str)
        return response.json({'ret_code': 1, 'return': {'content': {'pending_orders': pending_orders.to_dict('records')}}})

    @web_expt()
    async def get_completed_orders(self, request):
        start_date = request.args.get('start_date')
        completed_orders = self._algo.completed_orders
        if completed_orders.shape[0] > 0:
            if start_date is not None:
                completed_orders = completed_orders.loc[completed_orders['updated_time'] >= start_date]
        else:
            completed_orders = pd.DataFrame()
        return response.json(
            {'ret_code': 1, 'return': {'content': {'completed_orders': completed_orders.to_dict('records')}}})

    @web_expt()
    async def web_subscribe_tickers(self, request):
        tickers = eval(request.args.get('tickers'))
        new_tickers = list(set(tickers).difference(self._algo.universe))
        self._algo.subscribe_tickers(tickers=new_tickers)
        return response.json({'ret_code': 1, 'return': {'content': {'universe': list(self._algo.universe),
                                                                    'datatypes': list(self._algo.datatypes)}}})

    @web_expt()
    async def unsubscribe_ticker(self, request):
        tickers = eval(request.args.get('tickers'))
        self._algo.unsubscribe_tickers(tickers=tickers)
        return response.json({'ret_code': 1, 'return': {'content': {'universe': list(self._algo.universe),
                                                                    'datatypes': list(self._algo.datatypes)}}})

    @web_expt()
    async def pause(self, request):
        self._algo.pause()
        return response.json({'ret_code': 1, 'return': {'content': {'running': self._algo.running}}})

    @web_expt()
    async def resume(self, request):
        if self._algo.initialized:
            self._algo.resume()
        return response.json({'ret_code': 1, 'return': {'content': {'running': self._algo.running}}})

    def app_add_route(self, app):
        app.add_route(self.get_records, '/curves', methods=['GET'])
        app.add_route(self.get_positions, '/positions', methods=['GET'])
        app.add_route(self.get_pending_orders, '/pending', methods=['GET'])
        app.add_route(self.get_completed_orders, '/completed', methods=['GET'])
        app.add_route(self.get_attributes, '/attributes', methods=['GET'])
        app.add_route(self.get_summary, '/summary', methods=['GET'])
        app.add_route(self.web_subscribe_tickers, '/subscribe', methods=['POST'])
        app.add_route(self.unsubscribe_ticker, '/unsubscribe', methods=['POST'])
        app.add_route(self.pause, '/pause', methods=['GET', 'SET', 'POST'])
        app.add_route(self.resume, '/resume', methods=['GET', 'SET', 'POST'])