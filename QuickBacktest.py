import pandas as pd
import numpy as np
import quantstats as qs

# def create_signal(df, strategy_func, states, signal_mode='qty', max_rows=None):
#     df = df.copy()
#     signal_list = list()
#     value_list = list()
#     max_rows = df.shape[0] if max_rows is None else max_rows
#     states = states
#     for i in range(0, df.shape[0]):
#         partial_df = df.iloc[:i + 1].iloc[-max_rows:]
#         signal, value = strategy_func(df=partial_df, states=states)
#         signal_list.append(signal)
#         value_list.append(value)
#     df['signal'] = signal_list
#     df[signal_mode] = value_list
#     return df


class BacktestResult:
    def __init__(self, result_df, trade_df):
        self.result_df = result_df
        self.trade_df = trade_df

    def report(self, benchmark):
        import webbrowser

        PV = self.result_df.set_index('datetime')['pv']
        html = 'backtest result.html'
        qs.reports.html(PV, benchmark, output=html, title=f'Backtest result')
        webbrowser.open(html)

    def plot(self):
        df = self.result_df.copy()
        df['trade_qty'] = df['quantity'].diff()
        df['buy_pt'] = [1 if qty_dif > 0 else None for qty_dif in df['trade_qty']]
        df['sell_pt'] = [1 if qty_dif < 0 else None for qty_dif in df['trade_qty']]
        df['buy_y'] = df['buy_pt'] * df['close']
        df['sell_y'] = df['sell_pt'] * df['close']
        df['x'] = df['datetime']
        from matplotlib import pyplot as plt
        plt.scatter(x=df['x'], y=df['buy_y'].values, marker='o', color='green', s=100)
        plt.scatter(x=df['x'], y=df['sell_y'].values, marker='o', color='red', s=100)
        plt.ylabel(f'price')
        plt.plot(df['x'], df['close'])
        plt.title(f'Entry-exit points')

    def stats(self):
        stats = dict()
        pv = self.result_df.set_index('datetime')['pv']
        tk = self.result_df['adjclose'] if 'adjclose' in self.result_df.columns else self.result_df['close']
        ret = np.log(pv).diff()
        tk_ret = np.log(tk).diff()
        stats['N'] = pv.shape[0] - 1
        stats['Sharpe'] = qs.stats.sharpe(pv)
        stats['Sortino'] = qs.stats.sortino(pv)
        stats['CAGR %'] = (qs.stats.cagr(pv)) * 100
        stats['Cum Return %'] = (pv.iloc[-1] / pv.iloc[0] - 1) * 100
        stats['Cum Return (Stock) %'] = (tk.iloc[-1] / tk.iloc[0] - 1) * 100
        stats['Daily Ret %'] = (qs.stats.avg_return(pv)) * 100
        stats['Daily Vol %'] = (qs.stats.volatility(pv) / (251 ** (1 / 2))) * 100
        stats['Monthly Ret %'] = (stats['Daily Ret %'] * 21)
        stats['Monthly Vol %'] = (qs.stats.volatility(pv) / (21 ** (1 / 2))) * 100
        stats['Annual Ret %'] = (stats['Daily Ret %'] * 251)
        stats['Annual Vol %'] = (qs.stats.volatility(pv)) * 100
        stats['Win Days %'] = (qs.stats.win_rate(pv)) * 100
        stats['Max Drawdown %'] = (qs.stats.max_drawdown(pv)) * 100
        stats['Daily VAR %'] = (qs.stats.var(pv)) * 100
        stats['Beta'] = np.cov(ret.dropna(), tk_ret.dropna())[0][1] / np.var(tk_ret.dropna())
        stats['Alpha'] = stats['Cum Return %'] - stats['Cum Return (Stock) %']
        stats['No Trades'] = self.trade_df.shape[0]
        return stats


class BacktestResults:
    def __init__(self, backtest_results: dict):
        self.all_results = backtest_results

    def ticker_report(self, ticker, benchmark):
        self.all_results[ticker].report(benchmark=benchmark)

    def ticker_plot(self, ticker):
        self.all_results[ticker].plot()

    def ticker_stats(self, ticker):
        return self.all_results[ticker].stats()

    def stats(self):
        tk_stats = dict()
        for tk in self.all_results.keys():
            tk_stats[tk] = self.all_results[tk].stats()
        return np.round(pd.DataFrame(tk_stats), 2)

    def portfolio_report(self, benchmark, allocations=None):
        import webbrowser
        portfolio_pv = self.make_portfolio(allocations=allocations)['pv']
        html = 'backtest result.html'
        qs.reports.html(portfolio_pv, benchmark, output=html, title=f'Backtest result')
        webbrowser.open(html)

    def make_portfolio(self, allocations=None):
        df = pd.concat([x.result_df.set_index('datetime')['pv'] for x in self.all_results.values()], axis=1)
        df.columns = self.all_results.keys()
        df = df.fillna(method='bfill')
        allocations = [1 / len(self.all_results)] * len(self.all_results) if allocations is None else allocations
        return pd.DataFrame({'datetime': df.index, 'pv': np.dot(df, allocations)}).set_index('datetime')

def run_backtest(df, strategy_func, capital, states, buy_at_open=True, bid_ask_spread: float = 0.0,
                 fee_mode: str='FIXED:0', max_rows = None):

    records = pd.DataFrame(columns=['cash', 'quantity'])
    records.index.name = 'datetime'
    trades = pd.DataFrame(columns=['trade side', 'trade price', 'trade quantity', 'trade proceeds', 'transaction fee'])
    trades.index.name = 'datetime'

    cur_qty = 0
    cur_cash = capital

    df.columns = [x.lower() for x in df.columns]
    df['next open'] = df['open'].shift(-1)
    df['next datetime'] = df['datetime'].shift(-1)
    df.reset_index(drop=True, inplace=True)

    last_id = df.index[-1]
    action_price_label = 'next open' if buy_at_open else 'close'
    action_datetime = 'next datetime' if buy_at_open else 'datetime'

    try:
        mode, fee = fee_mode.split(':')
        fee = float(fee)
        if mode.upper() == 'FIXED':
            def cal_fee(qty, price):
                return fee
        elif mode.upper() == 'QTY':
            def cal_fee(qty, price):
                return abs(int(qty)) * fee
        elif mode.upper() == 'PERCENT':
            def cal_fee(qty, price):
                return abs(int(qty)) * price * fee
        else:
            raise Exception()

    except Exception:
        print(f'Invalid fee mode {fee_mode}, using zero fees')

        def cal_fee(qty, price):
            return 0.0

    max_rows = df.shape[0] if max_rows is None else max_rows
    for id, row in df.iterrows():
        if buy_at_open:
            records.loc[row['datetime']] = [cur_cash, cur_qty]

        if (id == last_id) and buy_at_open:
            continue

        action_price = row[action_price_label]
        action_dt = row[action_datetime]

        states['cash'] = cur_cash
        states['quantity'] = cur_qty
        states['trade price'] = action_price
        states['trade date'] = action_dt

        partial_df = df.iloc[:id + 1].iloc[-max_rows:].copy()
        signal, input_qty = strategy_func(df=partial_df, states=states)

        signal = signal.upper()
        input_qty = str(input_qty).upper()

        if signal == 'PASS':
            pass

        elif signal == 'LONG' and cur_cash > 0:
            action_price = action_price * (1 + bid_ask_spread)
            max_qty = int(cur_cash / action_price)
            bot_qty = max_qty if input_qty == 'ALL' else min(max_qty, abs(int(input_qty)))

            cash_proceeds = -bot_qty * action_price
            txn_fee = cal_fee(bot_qty, action_price)

            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty += bot_qty

            trades.loc[action_dt] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

        elif signal == 'EXIT LONG' and cur_qty > 0:
            action_price = action_price * (1 - bid_ask_spread)

            exit_qty = cur_qty

            cash_proceeds = cur_qty * action_price
            txn_fee = cal_fee(exit_qty, action_price)

            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty = 0
            trades.loc[action_dt] = ['SELL', action_price, exit_qty, cash_proceeds, txn_fee]

        elif signal == 'SHORT':
            action_price = action_price * (1 - bid_ask_spread)

            ev = cur_qty * action_price
            pv = cur_cash + ev
            min_short = -int(pv / action_price)

            short_qty = (cur_qty - min_short) if input_qty == 'ALL' else abs(int(input_qty))
            if cur_qty - short_qty < min_short:
                short_qty = cur_qty - min_short

            cash_proceeds = short_qty * action_price
            txn_fee = cal_fee(short_qty, action_price)
            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty -= short_qty
            trades.loc[action_dt] = ['SELL', action_price, short_qty, cash_proceeds, txn_fee]

        elif signal == 'COVER SHORT' and cur_qty < 0:
            action_price = action_price * (1 + bid_ask_spread)

            bot_qty = cur_qty * -1

            cash_proceeds = cur_qty * action_price
            txn_fee = cal_fee(bot_qty, action_price)
            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty += bot_qty
            trades.loc[action_dt] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

        elif signal == 'COVER AND LONG' and cur_qty <= 0:
            action_price = action_price * (1 + bid_ask_spread)

            pv = cur_qty * action_price + cur_cash
            max_qty = int(pv / action_price)
            long_qty = max_qty if input_qty == 'ALL' else min(max_qty, int(input_qty))
            bot_qty = -1 * cur_qty + long_qty

            cash_proceeds = -action_price * bot_qty
            txn_fee = cal_fee(bot_qty, action_price)

            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty += bot_qty
            trades.loc[action_dt] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

        elif signal == 'EXIT AND SHORT' and cur_qty >= 0:
            action_price = action_price * (1 - bid_ask_spread)

            pv = cur_qty * action_price + cur_cash
            max_qty = int(pv / action_price)
            short_qty = max_qty if input_qty == 'ALL' else min(max_qty, abs(int(input_qty)))
            sell_qty = -short_qty - cur_qty

            cash_proceeds = - sell_qty * action_price
            txn_fee = cal_fee(sell_qty, action_price)
            cur_cash = cur_cash + cash_proceeds - txn_fee

            cur_qty += sell_qty
            trades.loc[action_dt] = ['SELL', action_price, -sell_qty, cash_proceeds, txn_fee]

        elif signal == 'LIQUIDATE' and cur_qty != 0:
            if cur_qty > 0:
                action_price = action_price * (1 - bid_ask_spread)
            else:
                action_price = action_price * (1 + bid_ask_spread)

            action_qty = -cur_qty
            cash_proceeds = cur_qty * action_price

            txn_fee = cal_fee(action_qty, action_price)
            cur_cash = cur_cash + cash_proceeds - txn_fee

            cur_qty = 0
            trd_side = 'SELL' if action_qty < 0 else 'BUY'
            trades.loc[action_dt] = [trd_side, action_price, abs(action_qty), cash_proceeds, txn_fee]
        else:
            print(f'Invalid signal {signal} at {row["datetime"]} with qty {cur_qty}, passed without actions')

        if not buy_at_open:
            records.loc[row['datetime']] = [cur_cash, cur_qty]

    completed_df = df.copy()
    completed_df = completed_df.merge(records.reset_index(), on=['datetime'], how='right')
    completed_df['ev'] = (completed_df['quantity'] * completed_df['close']).astype(float)
    completed_df['pv'] = (completed_df['ev'] + completed_df['cash']).astype(float)
    completed_df['interval return'] = np.log(completed_df['pv']).diff()
    completed_df['datetime'] = pd.to_datetime(completed_df['datetime'])
    trades = trades.reset_index()
    trades['datetime'] = pd.to_datetime(trades['datetime'])
    return BacktestResult(result_df=completed_df, trade_df=trades)


def backtest(tickers, strategy_func, init_capital=1000000, buy_at_open=True,
             bid_ask_spread: float = 0.0,
             fee_mode: str = 'FIXED:0', start_date=None, end_date=None,
             max_rows=None, states: dict=None):
    backtest_result_dict = dict()
    print('Downloading data from Yahoo...')
    dfs = yh.Stocks(tickers).prices()
    for tk in tickers:
        if tk not in dfs.keys():
            print(f'Pass backtesting of {tk}...')
            continue
        df = dfs[tk]
        if start_date:
            df = df.loc[df['datetime'] >= start_date]
        if end_date:
            df = df.loc[df['datetime'] <= end_date]

        states_copy = dict() if states is None else states.copy()

        # df = create_signal(df=df, strategy_func=strategy_func, max_rows=max_rows, signal_mode=signal_mode, states=states_copy)
        print(f'Backtesting {tk}...')
        backtest_result = run_backtest(df, strategy_func=strategy_func, states=states_copy,
                                       capital=init_capital, buy_at_open=buy_at_open, bid_ask_spread=bid_ask_spread,
                                       fee_mode=fee_mode)
        backtest_result_dict[tk] = backtest_result
    print('Completed!')
    return BacktestResults(backtest_results=backtest_result_dict)

def convert_percent_to_qty(percentage, cur_cash, cur_qty, trade_price):
    if percentage == '':
        return "PASS", 0
    ev = cur_qty * trade_price
    pv = cur_cash + ev
    try:
        percent = float(percentage)
        if percent > 1 or percent < -1:
            raise Exception()
    except Exception:
        print(f'Invalid % invested {percentage}, return PASS signal')
        return "PASS", 0

    required_qty = int(pv * percent / trade_price)
    action_qty = required_qty - cur_qty
    if action_qty == 0:
        return "PASS", 0
    if action_qty > 0:
        return "LONG", action_qty
    else:
        return "SHORT", abs(action_qty)

if __name__ == '__main__':
    import Yahoo as yh


    # df = yh.Stock('0700.HK').price()
    # df = df.loc[df['datetime'] >= '2015-01-02']
    #
    #
    # df = create_signal(df=df, strategy_func=sma_crossover_signal, max_rows=33)
    # backtest_result = run_backtest(df, capital=100000, buy_at_open=False)
    #
    #
    # df = yh.Stock('1299.HK').price()
    # df = df.loc[df['datetime'] >= '2015-01-02']
    #
    # def sma_crossover_signal(df):
    #     df['sma16'] = df['adjclose'].rolling(16).mean()
    #     df['sma32'] = df['adjclose'].rolling(32).mean()
    #     df['dif'] = df['sma16'] - df['sma32']
    #     df['pre_dif'] = df['dif'].shift(1)
    #     row = df.iloc[-1]
    #     if row['dif'] > 0 and row['pre_dif'] <= 0:
    #         return 'COVER AND LONG', 'ALL'
    #
    #     elif row['dif'] < 0 and row['pre_dif'] >= 0:
    #         return 'EXIT AND SHORT', 'ALL'
    #     else:
    #         return 'PASS', ''
    #
    # df = create_signal(df=df, strategy_func=sma_crossover_signal, max_rows=33)
    # backtest_result2 = run_backtest(df, capital=100000, buy_at_open=False)

    # def sma_crossover_signal(df):
    #     df['sma16'] = df['adjclose'].rolling(16).mean()
    #     df['sma32'] = df['adjclose'].rolling(32).mean()
    #     df['dif'] = df['sma16'] - df['sma32']
    #     df['pre_dif'] = df['dif'].shift(1)
    #     row = df.iloc[-1]
    #     if row['dif'] > 0 and row['pre_dif'] <= 0:
    #         return 'LONG', 'ALL'
    #
    #     elif row['dif'] < 0 and row['pre_dif'] >= 0:
    #         return 'EXIT LONG', 'ALL'
    #     else:
    #         return 'PASS', ''

    import random

    def sma_crossover_signal(df, states):
        df['sma16'] = df['adjclose'].rolling(16).mean()
        df['sma32'] = df['adjclose'].rolling(32).mean()
        df['dif'] = df['sma16'] - df['sma32']
        df['pre_dif'] = df['dif'].shift(1)
        row = df.iloc[-1]
        print(states['cash'])
        if row['dif'] > 0 and row['pre_dif'] <= 0:
            return 'LONG', 'ALL'

        elif row['dif'] < 0 and row['pre_dif'] >= 0:
            return 'EXIT LONG', 'ALL'
        else:
            return 'PASS', ''

    tickers = ['FB', 'AMZN', 'AAPL', 'GOOG', 'NFLX', 'MDB', 'NET', 'TEAM', 'CRM']
    result = backtest(tickers=tickers, strategy_func=sma_crossover_signal, start_date="2015-01-01",
                      end_date="2020-07-31", states={'id': 0})

    # if "allocations" is not specified, default equal weightings
    result.portfolio_report(benchmark="^IXIC")

    # # stats of all tickers
    # result.stats()
    #
    # # plot exit entry points of a ticker
    # result.ticker_plot('FB')
    #
    # # report of a ticker
    # result.ticker_report('FB', benchmark='^IXIC')