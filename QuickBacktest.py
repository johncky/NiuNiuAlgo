import pandas as pd
import numpy as np
from examples import research_tools
import quantstats as qs

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



def backtest(df, capital, fee_mode: str='FIXED:0', buy_at_open=True, spread: float = 0.0):
    # Very quick backtesting function, give it a df that contains columns: datetime, open, close, signal, qty
    # signal column should contain: 'PASS, LONG, EXIT LONG, SHORT, COVER SHORT, COVER AND LONG, EXIT AND SHORT, LIQUIDATE'
    # qty: INT type quantity to trade, or "ALL"
    # fee_mode: str: FIXED:FLOAT or QTY:FLOAT

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
    df['qty'] = df['qty'].astype(str).str.upper()

    last_id = df.index[-1]
    action_price_label = 'next open' if buy_at_open else 'close'
    action_datetime = 'next datetime' if buy_at_open else 'datetime'

    try:
        mode, fee = fee_mode.split(':')
        fee = float(fee)
        if mode.upper() == 'FIXED':
            def cal_fee(qty):
                return fee
        elif mode.upper() == 'QTY':
            def cal_fee(qty):
                return abs(int(qty)) * fee
        else:
            raise Exception()

    except Exception:
        print(f'Invalid fee mode {fee_mode}, using zero fees')
        def cal_fee(qty):
            return 0.0

    for id, row in df.iterrows():

        if buy_at_open:
            records.loc[row['datetime']] = [cur_cash, cur_qty]

        if (id == last_id) and buy_at_open:
            continue

        signal = row['signal'].upper()
        action_price = row[action_price_label]
        action_dt = row[action_datetime]
        if signal == 'PASS':
            continue

        elif signal == 'LONG' and cur_cash > 0:
            action_price = action_price * (1 + spread)
            max_qty = int(cur_cash / action_price)
            bot_qty = max_qty if row['qty'] == 'ALL' else min(max_qty, abs(int(row['qty'])))

            cash_proceeds = -bot_qty * action_price
            txn_fee = cal_fee(bot_qty)

            cur_cash = cur_cash + cash_proceeds  - txn_fee
            cur_qty += bot_qty

            trades.loc[action_dt] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

        elif signal == 'EXIT LONG' and cur_qty > 0:
            action_price = action_price * (1 - spread)

            exit_qty = cur_qty

            cash_proceeds = cur_qty * action_price
            txn_fee = cal_fee(exit_qty)

            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty = 0
            trades.loc[action_dt] = ['SELL', action_price, exit_qty, cash_proceeds, txn_fee]

        elif signal == 'SHORT' and cur_qty <= 0:
            action_price = action_price * (1 - spread)

            ev = cur_qty * action_price
            pv = cur_cash + ev
            max_qty = int(pv / action_price) + cur_qty
            short_qty = max_qty if row['qty'] == 'ALL' else min(abs(int(row['qty'])), max_qty)

            cash_proceeds = short_qty * action_price
            txn_fee = cal_fee(short_qty)
            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty -= short_qty
            trades.loc[action_dt] = ['SELL', action_price, short_qty, cash_proceeds, txn_fee]

        elif signal == 'COVER SHORT' and cur_qty < 0:
            action_price = action_price * (1 + spread)

            bot_qty = cur_qty * -1

            cash_proceeds = - bot_qty * action_price
            txn_fee = cal_fee(bot_qty)
            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty += bot_qty
            trades.loc[action_dt] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

        elif signal == 'COVER AND LONG' and cur_qty <= 0:
            action_price = action_price * (1 + spread)

            pv = cur_qty * action_price + cur_cash
            max_qty = int(pv / action_price)
            long_qty = max_qty if row['qty'] == 'ALL' else min(max_qty, int(row['qty']))
            bot_qty = -1 * cur_qty + long_qty

            cash_proceeds = -action_price * bot_qty
            txn_fee = cal_fee(bot_qty)

            cur_cash = cur_cash + cash_proceeds - txn_fee
            cur_qty += bot_qty
            trades.loc[action_dt] = ['BUY', action_price, bot_qty, cash_proceeds, txn_fee]

        elif signal == 'EXIT AND SHORT' and cur_qty >= 0:
            action_price = action_price * (1 - spread)

            pv = cur_qty * action_price + cur_cash
            max_qty = int(pv / action_price)
            short_qty = max_qty if row['qty'] == 'ALL' else min(max_qty, int(row['qty']))
            sell_qty = -short_qty - cur_qty

            cash_proceeds = - sell_qty * action_price
            txn_fee = cal_fee(sell_qty)
            cur_cash = cur_cash + cash_proceeds - txn_fee

            cur_qty += sell_qty
            trades.loc[action_dt] = ['SELL', action_price, -sell_qty, cash_proceeds, txn_fee]

        elif signal == 'LIQUIDATE' and cur_qty != 0:
            if cur_qty > 0:
                action_price = action_price * (1 - spread)
            else:
                action_price = action_price * (1 + spread)

            action_qty = -cur_qty
            cash_proceeds = cur_qty * action_price

            txn_fee = cal_fee(action_qty)
            cur_cash = cur_cash + cash_proceeds - txn_fee

            cur_qty = 0
            trd_side = 'SELL' if action_qty < 0 else 'BUY'
            trades.loc[action_dt] = [trd_side, action_price, abs(action_qty), cash_proceeds, txn_fee]
        else:
            print(f'Invalid signal {signal} at {row["datetime"]}, passed without actions')

        if not buy_at_open:
            records.loc[row['datetime']] = [cur_cash, cur_qty]


    completed_df = df.copy()
    completed_df = completed_df.merge(records.reset_index(), on=['datetime'], how='right')
    completed_df['ev'] = completed_df['quantity'] * completed_df['close']
    completed_df['pv'] = completed_df['ev'] + completed_df['cash']
    completed_df['ev'] = completed_df['ev'].astype(float)
    completed_df['pv'] = completed_df['pv'].astype(float)
    completed_df['interval return'] = np.log(completed_df['pv']).diff()
    completed_df['datetime'] = pd.to_datetime(completed_df['datetime'])
    trades = trades.reset_index()
    trades['datetime'] = pd.to_datetime(trades['datetime'])
    return BacktestResult(result_df=completed_df, trade_df=trades)


if __name__ == '__main__':
    df = research_tools.download_data('K_DAY', 'HK.00700')
    df['sma16'] = df['close'].rolling(16).mean()
    df['sma32'] = df['close'].rolling(32).mean()
    df['dif'] = df['sma16'] - df['sma32']
    df['pre_dif'] = df['dif'].shift(1)
    df = df.dropna()
    signal = list()
    for id, row in df.iterrows():
        if row['dif'] > 0 and row['pre_dif'] <= 0:
            signal.append('COVER AND LONG')
        elif row['dif'] < 0 and row['pre_dif'] >= 0:
            signal.append('EXIT AND SHORT')
        else:
            signal.append('PASS')
    df['signal'] = signal
    df['qty'] = 'ALL'
    backtest_result = backtest(df, capital=100000)