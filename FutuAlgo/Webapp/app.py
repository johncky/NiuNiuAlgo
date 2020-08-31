from sanic import Sanic
from jinja2 import Template
import asyncio
import json
from sanic import response
import collections
import pandas as pd
import datetime
import aiohttp
from aiohttp import ClientConnectionError
import math
import random

BENCHMARK_TICKER = {'HSI': 'HK.800000', 'SPX': 'HK.800000'}


class WebApp:
    def __init__(self, max_curve_rows=10000):
        self.app = Sanic('Dashboard')
        self.app_add_route(app=self.app)

        self.hook_ip = None

        self.algo_ips = dict()
        self.algo_data = collections.defaultdict(lambda: collections.defaultdict(lambda: pd.DataFrame()))
        self.algo_curves = collections.defaultdict(lambda: collections.defaultdict(lambda: pd.DataFrame()))
        self.failed_algo = dict()

        self.benchmark_df = collections.defaultdict(lambda: pd.DataFrame())

        self.last_update_time = None
        self.max_curve_rows = max_curve_rows
        self.port = None

    async def update_summary(self, algo_name):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.algo_ips[algo_name] + '/summary') as resp:
                    result = await resp.json()

            resp_df = pd.DataFrame(result['return']['content'], index=[0])
            self.algo_data[algo_name]['summary'] = self.algo_data[algo_name]['summary'].append(resp_df).drop_duplicates(
                ['name'])
        except ClientConnectionError:
            self.failed_algo[algo_name] = self.algo_ips[algo_name]
            del self.algo_ips[algo_name]
            self.algo_data[algo_name] = collections.defaultdict(lambda: pd.DataFrame())
            raise

    async def update_curves(self, algo_name):
        if self.algo_curves[algo_name]['PV'].shape[0] == 0:
            start_date = '2000-01-01'
        else:
            start_date = min(self.algo_curves[algo_name]['PV']['x']).strftime('%Y-%m-%d')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.algo_ips[algo_name] + '/curves', params={'start_date': start_date}) as resp:
                    result = await resp.json()

            result = result['return']['content']
            for curve_type in result.keys():
                tmp_df = pd.DataFrame(result[curve_type], index=[0]) if len(result[curve_type]) == 1 else pd.DataFrame(
                    result[curve_type])
                tmp_df['x'] = pd.to_datetime(tmp_df['x'])
                self.algo_curves[algo_name][curve_type] = self.algo_curves[algo_name][curve_type].append(tmp_df)
                self.algo_curves[algo_name][curve_type] = self.algo_curves[algo_name][curve_type].drop_duplicates(['x'])

                if self.algo_curves[algo_name][curve_type].shape[0] >= self.max_curve_rows:
                    self.algo_curves[algo_name][curve_type] = self.algo_curves[algo_name][curve_type].iloc[
                                                              -self.max_curve_rows:]
        except ClientConnectionError:
            self.failed_algo[algo_name] = self.algo_ips[algo_name]
            del self.algo_ips[algo_name]
            self.algo_curves[algo_name] = collections.defaultdict(lambda: pd.DataFrame())
            raise

    async def update_positions(self, algo_name):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.algo_ips[algo_name] + '/positions') as resp:
                    result = await resp.json()

            resp_df = pd.DataFrame(result['return']['content']['positions'], index=[0]) if len(
                result) == 1 else pd.DataFrame(
                result['return']['content']['positions'])
            self.algo_data[algo_name]['positions'] = resp_df

        except ClientConnectionError:
            self.failed_algo[algo_name] = self.algo_ips[algo_name]
            del self.algo_ips[algo_name]
            self.algo_data[algo_name] = collections.defaultdict(lambda: pd.DataFrame())
            raise

    async def update_settings(self, algo_name):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.algo_ips[algo_name] + '/attributes') as resp:
                    result = await resp.json()
            self.algo_data[algo_name]['settings'] = result['return']['content']

        except ClientConnectionError:
            self.failed_algo[algo_name] = self.algo_ips[algo_name]
            del self.algo_ips[algo_name]
            self.algo_data[algo_name] = collections.defaultdict(lambda: pd.DataFrame())
            raise

    async def update_pending(self, algo_name):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.algo_ips[algo_name] + '/pending') as resp:
                    result = await resp.json()
            resp_df = pd.DataFrame(result['return']['content']['pending_orders'], index=[0]) if len(
                result) == 1 else pd.DataFrame(
                result['return']['content']['pending_orders'])
            self.algo_data[algo_name]['pending'] = resp_df

        except ClientConnectionError:
            self.failed_algo[algo_name] = self.algo_ips[algo_name]
            del self.algo_ips[algo_name]
            self.algo_data[algo_name] = collections.defaultdict(lambda: pd.DataFrame())
            raise

    async def update_completed(self, algo_name):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.algo_ips[algo_name] + '/completed') as resp:
                    result = await resp.json()

            resp_df = pd.DataFrame(result['return']['content']['completed_orders'], index=[0]) if len(
                result) == 1 else pd.DataFrame(
                result['return']['content']['completed_orders'])
            self.algo_data[algo_name]['completed'] = resp_df

        except ClientConnectionError:
            self.failed_algo[algo_name] = self.algo_ips[algo_name]
            del self.algo_ips[algo_name]
            self.algo_data[algo_name] = collections.defaultdict(lambda: pd.DataFrame())
            raise

    async def update_benchmark_data(self):
        earliest_deployment_date = '2000-01-01'

        for algo_name in self.algo_ips.keys():
            earliest_deployment_date = min(earliest_deployment_date,
                                           self.algo_data[algo_name]['summary']['initialized_date'][0])

        for index in BENCHMARK_TICKER.keys():
            if self.benchmark_df[index].shape[0] == 0:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.hook_ip + '/historicals',
                                           params={'ticker': BENCHMARK_TICKER[index], 'datatype': 'K_DAY',
                                                   'start_date': earliest_deployment_date,
                                                   'from_exchange': 'false'}) as resp:
                        result = await resp.json()

                self.benchmark_df[index] = pd.read_json(result['return']['content'])

            else:
                start_date = self.benchmark_df[index]['datetime'].iloc[-1].strftime('%Y-%m-%d')
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.hook_ip + '/historicals',
                                           params={'ticker': BENCHMARK_TICKER[index], 'datatype': 'K_DAY',
                                                   'start_date': start_date,
                                                   'from_exchange': 'false'}) as resp:
                        result = await resp.json()

                tmp_df = pd.read_json(result['return']['content'])
                self.benchmark_df[index] = self.benchmark_df[index].append(tmp_df).drop_duplicates(['datetime'])

    # --------------------------------------- Return calculations ---------------------------------------------------
    @staticmethod
    def get_pnl_pct(df, start_date):
        if 'x' in df.columns:
            x = 'x'
            y = 'y'

        else:
            if 'datetime' not in df.columns:
                return 0, 0
            x = 'datetime'
            y = 'close'

        df = df.loc[df[x] >= start_date]
        if df.shape[0] == 0:
            return 0, 0
        pnl_pct = df[y].iloc[-1] / df[y].iloc[0] - 1
        years = (df[x].iloc[-1] - df[x].iloc[0]).days / 365
        annualized_pct = (1 + pnl_pct) ** (1 / years) - 1
        return pnl_pct, annualized_pct

    @staticmethod
    def get_returns(df, start_date=None):
        if 'x' in df.columns:
            x = 'x'
            y = 'y'

        else:
            if 'datetime' not in df.columns:
                return (0, 0), (0, 0)
            x = 'datetime'
            y = 'close'

        if start_date:
            df = df.loc[df[x] >= start_date]

        if df.shape[0] == 0:
            return 0, 0

        df[x] = pd.to_datetime(df[x])
        df = df.set_index(x)

        d_df = df.resample('D').last().dropna()
        ytd_pv = d_df[y].iloc[max(-2, -d_df.shape[0])]
        d_pct = d_df[y].iloc[-1] / ytd_pv - 1

        d_return = d_pct * ytd_pv

        m_df = df.resample('M').last()
        last_month_pv = m_df[y].iloc[max(-2, -m_df.shape[0])]
        m_pct = m_df[y].iloc[-1] / last_month_pv - 1
        m_return = m_pct * last_month_pv

        return (d_return, d_pct), (m_return, m_pct)

    @staticmethod
    def calc_returns(pv_df, benchmark_df):
        pv_df['x'] = pd.to_datetime(pd.to_datetime(pv_df['x']).dt.strftime('%Y-%m-%d'))
        pv_df = pv_df.set_index('x').resample('D').last().reset_index()
        benchmark_df = benchmark_df.rename(columns={'datetime': 'x'})
        pv_bmk_df = pv_df.merge(benchmark_df, how='right', on=['x']).dropna()[['x', 'y', 'close']].set_index('x')
        if pv_bmk_df.shape[0] <= 1:
            return (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0,0,0)

        d_return = pv_bmk_df['y'].iloc[-1] - pv_bmk_df['y'].iloc[-2]
        d_pct = d_return / pv_bmk_df['y'].iloc[-2]
        bmk_pct = pv_bmk_df['close'].iloc[-1] / pv_bmk_df['close'].iloc[-2] - 1
        bmk_return = pv_bmk_df['y'].iloc[-2] * bmk_pct

        pv_bmk_m_df = pv_bmk_df.resample('M').last()
        if pv_bmk_m_df.shape[0] == 1:
            m_return = pv_bmk_df['y'].iloc[-1] - pv_bmk_df['y'].iloc[0]
            m_pct = m_return / pv_bmk_df['y'].iloc[0]
            bmk_m_pct = pv_bmk_df['close'].iloc[-1] / pv_bmk_df['close'].iloc[0] - 1
            bmk_m_return = bmk_m_pct * pv_bmk_df['y'].iloc[0]

        else:
            m_return = pv_bmk_m_df['y'].iloc[-1] - pv_bmk_m_df['y'].iloc[-2]
            m_pct = m_return / pv_bmk_m_df['y'].iloc[-2]
            bmk_m_pct = pv_bmk_m_df['close'].iloc[-1] / pv_bmk_m_df['close'].iloc[0] - 1
            bmk_m_return = bmk_m_pct * pv_bmk_m_df['y'].iloc[0]

        total_pnl = pv_bmk_df['y'].iloc[-1] - pv_bmk_df['y'].iloc[0]
        total_pct = total_pnl / pv_bmk_df['y'].iloc[0]
        total_bmk_pct = pv_bmk_df['close'].iloc[-1] / pv_bmk_df['close'].iloc[0] - 1
        total_bmk_pnl = total_bmk_pct * pv_bmk_df['y'].iloc[0]

        annualized_pct = (total_pct + 1) ** (365 / (pv_bmk_df.index[-1] - pv_bmk_df.index[0]).days)
        bmk_annualized_pct = (total_bmk_pct + 1) ** (365 / (pv_bmk_df.index[-1] - pv_bmk_df.index[0]).days)

        # Sharpe, Beta,
        # TODO: 3 months
        three_month_daily_df = pv_bmk_df.iloc[-min(pv_bmk_df.shape[0], 253):]
        three_month_daily_df['y_ret'] = three_month_daily_df['y'] / three_month_daily_df['y'].shift(1) - 1
        three_month_daily_df['bmk_ret'] = three_month_daily_df['close'] / three_month_daily_df['close'].shift(1) - 1
        three_month_daily_df = three_month_daily_df.dropna()
        if three_month_daily_df.shape[0] <= 1:
            beta = 0
            sharpe = 0
            benchmark_sharpe = 0
        else:
            three_month_daily_df = three_month_daily_df[['y_ret', 'bmk_ret']]
            beta = three_month_daily_df.cov().iloc[0, 1] / (three_month_daily_df['bmk_ret'].std() ** 2)
            sharpe = three_month_daily_df['y_ret'].mean() / three_month_daily_df['y_ret'].std() * (251 ** (1/2))
            benchmark_sharpe = three_month_daily_df['bmk_ret'].mean() / three_month_daily_df['bmk_ret'].std()
        return (d_return, d_pct), (m_return, m_pct), (bmk_return, bmk_pct), (bmk_m_return, bmk_m_pct), \
               (total_pct, total_bmk_pct), (total_pnl, total_bmk_pnl), (annualized_pct, bmk_annualized_pct), (sharpe, beta, benchmark_sharpe)

    async def update_returns(self, algo_name):
        await self.update_benchmark_data()
        benchmark_dfs = self.benchmark_df

        algo_benchmark = self.algo_data[algo_name]['summary']['benchmark'][0]
        pv_df = self.algo_curves[algo_name]['PV'].copy()
        bmk_df = benchmark_dfs[algo_benchmark]
        (d_return, d_pct), (m_return, m_pct), (bmk_return, bmk_pct), (bmk_m_return, bmk_m_pct), \
        (total_pct, total_bmk_pct), (total_pnl, total_bmk_pnl), (annualized_pct, bmk_annualized_pct), (sharpe, beta, benchmark_sharpe) = self.calc_returns(pv_df, bmk_df)

        self.algo_data[algo_name]['summary']['benchmark_net_pnl_pct'] = total_bmk_pct
        self.algo_data[algo_name]['summary']['benchmark_annualized_return'] = bmk_annualized_pct
        self.algo_data[algo_name]['summary']['daily_return'] = d_return
        self.algo_data[algo_name]['summary']['daily_return_pct'] = d_pct
        self.algo_data[algo_name]['summary']['monthly_return'] = m_return
        self.algo_data[algo_name]['summary']['monthly_return_pct'] = m_pct
        self.algo_data[algo_name]['summary']['benchmark_daily_pct'] = bmk_pct
        self.algo_data[algo_name]['summary']['benchmark_monthly_pct'] = bmk_m_pct
        self.algo_data[algo_name]['summary']['gross_pnl'] = total_pnl
        self.algo_data[algo_name]['summary']['gross_pnl_pct'] = total_pct
        self.algo_data[algo_name]['summary']['net_pnl'] = total_pnl - self.algo_data[algo_name]['summary']['txn_cost_total']
        self.algo_data[algo_name]['summary']['net_pnl_pct'] = self.algo_data[algo_name]['summary']['net_pnl'] / self.algo_data[algo_name]['summary']['initial_capital']
        self.algo_data[algo_name]['summary']['annualized_return'] = annualized_pct
        self.algo_data[algo_name]['summary']['sharpe'] = sharpe
        self.algo_data[algo_name]['summary']['beta'] = beta
        self.algo_data[algo_name]['summary']['benchmark_sharpe'] = benchmark_sharpe
        # sharpe

        # benchmark_sharpe
        # sortino
        # benchmark_sortino
        # win_pct
        # benchmark_win_pct
        pass

    def run(self, port, hook_ip):
        loop = asyncio.get_event_loop()
        self.port = port
        self.hook_ip = hook_ip

        async def _run():
            tasks = list()
            web_server = self.app.create_server(host='0.0.0.0', return_asyncio_server=True, port=port)
            tasks.append(web_server)
            await asyncio.gather(*tasks)

        loop.create_task(_run())
        loop.run_forever()

    # -------------------------------------------- WebApp ----------------------------------------------------------
    async def download_data_from_algos(self, algo_name):
        await self.update_summary(algo_name)
        await self.update_positions(algo_name)
        await self.update_pending(algo_name)
        await self.update_completed(algo_name)
        await self.update_settings(algo_name)
        await self.update_curves(algo_name)


    async def get_combined_data(self):
        # for algo_name in self.algo_ips.keys():
        #     await self.update_summary(algo_name)
        #     await self.update_positions(algo_name)
        #     await self.update_pending(algo_name)
        #     await self.update_completed(algo_name)
        #     await self.update_curves(algo_name)
        #     await self.update_benchmark_data()
        #     await self.update_returns(algo_name)

        curves = collections.defaultdict(lambda: list())
        data = collections.defaultdict(lambda: 0.0)
        pending_list = list()
        completed_list = list()
        positions_list = list()

        data['name'] = 'combined'
        # TODO: changeable
        data['benchmark'] = 'HSI'
        data['status'] = 'Running'
        data['ip'] = 'http://127.0.0.1:' + str(self.port)

        earliest_deployment_date = '2100-01-01'
        max_days_since_deployment = 0

        for algo_name in self.algo_ips.keys():
            # Curves
            max_days_since_deployment = max(max_days_since_deployment,
                                            int(self.algo_data[algo_name]['summary']['days_since_deployment'][0]))
            earliest_deployment_date = min(earliest_deployment_date,
                                           self.algo_data[algo_name]['summary']['initialized_date'][0])
            curves['PV'].append(self.algo_curves[algo_name]['PV'].set_index('x').rename(columns={'y': algo_name}))
            curves['EV'].append(self.algo_curves[algo_name]['EV'].set_index('x').rename(columns={'y': algo_name}))
            curves['Cash'].append(self.algo_curves[algo_name]['Cash'].set_index('x').rename(columns={'y': algo_name}))

            # Values
            algo_summary = self.algo_data[algo_name]['summary'].to_dict('records')[0]
            data['n_trades'] += algo_summary['n_trades']
            data['txn_cost_total'] += algo_summary['txn_cost_total']
            data['initial_capital'] += algo_summary['initial_capital']
            data['gross_pnl'] += algo_summary['gross_pnl']
            data['net_pnl'] += algo_summary['net_pnl']
            data['gross_pnl'] += algo_summary['gross_pnl']

            # Orders & Positions
            pending_list.append(self.algo_data[algo_name]['pending'])
            completed_list.append(self.algo_data[algo_name]['completed'])
            positions_list.append(self.algo_data[algo_name]['positions'])

        # Sums up curves for ALL strategies
        pv_df = pd.DataFrame(pd.concat(curves['PV'], axis=1).fillna(method='bfill').fillna(method='ffill').sum(axis=1),
                             columns=['y']).reset_index()
        pv_df.columns = ['x', 'y']
        pv_df = pv_df.drop_duplicates('x', keep='last')

        ev_df = pd.DataFrame(pd.concat(curves['EV'], axis=1).fillna(method='bfill').fillna(method='ffill').sum(axis=1),
                             columns=['y']).reset_index()
        ev_df.columns = ['x', 'y']
        ev_df = ev_df.drop_duplicates('x', keep='last')

        cash_df = pd.DataFrame(pd.concat(curves['Cash'], axis=1).fillna(method='bfill').fillna(method='ffill').sum(axis=1),
                               columns=['y']).reset_index()
        cash_df.columns = ['x', 'y']
        cash_df = cash_df.drop_duplicates('x', keep='last')


        # Returns of combined
        (d_return, d_pct), (m_return, m_pct), (bmk_return, bmk_pct), (bmk_m_return, bmk_m_pct), \
        (total_pct, total_bmk_pct), (total_pnl, total_bmk_pnl), (
        annualized_pct, bmk_annualized_pct), (sharpe, beta, benchmark_sharpe) = self.calc_returns(pv_df.copy(), self.benchmark_df[data['benchmark']])

        data['daily_return'] = d_return
        data['daily_return_pct'] = d_pct
        data['monthly_return'] = m_return
        data['monthly_return_pct'] = m_pct
        data['benchmark_net_pnl_pct'] = total_bmk_pct
        data['benchmark_annualized_return'] = bmk_annualized_pct
        data['benchmark_daily_pct'] = bmk_pct
        data['benchmark_monthly_pct'] = bmk_m_pct
        data['gross_pnl_pct'] = data['gross_pnl'] / data['initial_capital']
        data['net_pnl_pct'] = data['net_pnl'] / data['initial_capital']
        data['annualized_return'] = annualized_pct
        data['sharpe'] = sharpe
        data['beta'] = beta
        data['benchmark_sharpe'] = benchmark_sharpe
        data['pv'] = pv_df['y'].iloc[-1]
        data['cash'] = cash_df['y'].iloc[-1]

        data['initialized_date'] = 'None'
        data['days_since_deployment'] = 'None'

        # Curves of combined
        data['EV'] = ev_df.astype('str').to_dict('records')
        data['PV'] = pv_df.astype('str').to_dict('records')
        data['Cash'] = cash_df.astype('str').to_dict('records')
        data['settings'] = []

        # Orders & Positions of combined
        combined_positions = pd.concat(positions_list)
        if combined_positions.shape[0] > 0:
            combined_positions = combined_positions.sort_values('ticker')

        combined_pendings = pd.concat(pending_list)
        if combined_pendings.shape[0] > 0:
            combined_pendings = combined_pendings.sort_values('updated_time')

        combined_completed = pd.concat(completed_list)
        if combined_completed.shape[0] > 0:
            combined_completed = combined_completed.sort_values('updated_time')

        data['positions'] = combined_positions.to_dict('records')
        data['pending'] = combined_pendings.to_dict('records')
        data['completed'] = combined_completed.to_dict('records')
        # sharpe
        # Beta
        # benchmark_sharpe
        # sortino
        # benchmark_sortino
        # win_pct
        # benchmark_win_pct

        return data

    async def get_all_data(self):
        if len(self.algo_ips) != 0:
            data = dict()
            data['algos_data'] = dict()
            for algo_name in self.algo_ips.keys():
                await self.download_data_from_algos(algo_name)

            await self.update_benchmark_data()
            for algo_name in self.algo_ips.keys():
                await self.update_returns(algo_name)
                algo_data = dict()
                algo_data['EV'] = self.algo_curves[algo_name]['EV'].astype('str').to_dict('records')
                algo_data['PV'] = self.algo_curves[algo_name]['PV'].astype('str').to_dict('records')
                algo_data['Cash'] = self.algo_curves[algo_name]['Cash'].astype('str').to_dict('records')
                algo_data['settings'] = self.algo_data[algo_name]['settings']
                algo_data['positions'] = self.algo_data[algo_name]['positions'].to_dict('records')
                algo_data['pending'] = self.algo_data[algo_name]['pending'].to_dict('records')
                algo_data['completed'] = self.algo_data[algo_name]['completed'].to_dict('records')

                for key, value in self.algo_data[algo_name]['summary'].to_dict('records')[0].items():
                    algo_data[key] = value
                data['algos_data'][algo_name] = algo_data
            data['algos_data']['combined'] = await self.get_combined_data()
            data['algos_ip'] = self.algo_ips.copy()
            data['app_ip'] = 'http://127.0.0.1:'+str(self.port)
            return data
        else:
            return {'algos_data': {}, 'algos_ip': [], 'app_ip': 'http://127.0.0.1:'+str(self.port)}

    async def get_data(self, request):
        data = await self.get_all_data()
        return response.json(data)

    async def add_algo(self, request):
        ip = request.args.get('ip')
        if ip in self.algo_ips.values():
            return response.json({'response': 'Algo already added!'})
        else:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(ip + '/summary') as resp:
                        result = await resp.json()
                if result['ret_code'] == 1:
                    algo_name = result['return']['content']['name']
                    if algo_name in self.algo_ips.keys():
                        return response.json({'response': f'Duplicate Algo name: {algo_name}!'})
                    else:
                        self.algo_ips[algo_name] = ip
                    return response.json({'response': f'Added Algo: {algo_name} sucessfully with ip: {ip}!'})
            except ClientConnectionError as e:
                return response.json({'response': f'Failed to connect to Algo ip {ip}, reason: {str(e)}'})

    async def remove_algo(self, request):
        ip = request.args.get('ip')
        for algo_name, algo_ip in self.algo_ips.items():
            if algo_ip == ip:
                del self.algo_ips[algo_name]
                return response.json({'response': f'Removed Algo {algo_name} successfully with ip: {algo_ip}'})
        return response.json({'response': f'Cannot find Algo with ip {ip}'})

    async def index(self, request):
        algo_data = await self.get_all_data()
        data = json.dumps({'data': algo_data})

        with open('templates/index.html') as file:
            template = Template(file.read())

        t = response.html(template.render(data=data, url_for=self.app.url_for))
        return t

    def app_add_route(self, app):
        app.static('/static', './static')
        app.add_route(self.index, '/', methods=['GET'])
        app.add_route(self.get_data, '/data', methods=['GET'])
        app.add_route(self.add_algo, '/add_algo', methods=['GET'])
        app.add_route(self.remove_algo, '/remove_algo', methods=['GET'])


if __name__ == '__main__':
    app = WebApp()
    # name & sanic ip address of the running algo
    app.run(8522, hook_ip='http://127.0.0.1:8000')
