import requests
import pandas as pd
import datetime
import talib

def download_daily_data(ticker):
    return download_data(datatype='K_DAY', ticker=ticker)


def downlaod_multiple_daily(tickers):
    dfs = list()
    for tk in tickers:
        dfs.append(download_daily_data(tk))
    return pd.concat(dfs, axis=0, keys=tickers)


def download_data(datatype, ticker):
    url = "http://127.0.0.1:8000/historicals"
    params = {'datatype': datatype, 'ticker': ticker, 'from_exchange': False}
    result = requests.get(url, params=params).json()
    df = pd.read_json(result['return']['content'])
    return df


def download_multiple_data(datatype, tickers):
    dfs = list()
    for tk in tickers:
        try:
            new_df = download_data(datatype=datatype, ticker=tk)
            dfs.append(new_df)
        except Exception:
            print(f'Failed to download {tk}')
    return pd.concat(dfs, axis=0, keys=tickers) if len(dfs) > 0 else pd.DataFrame()


def fill_daily_data(ticker):
    return fill_data(datatype='K_DAY', ticker=ticker)


def fill_multiple_daily(tickers):
    return fill_multiple_data(datatypes='K_DAY', tickers=tickers)


def fill_data(datatype, ticker):
    url = "http://192.168.3.3:8000/db/fill"
    params = {'ticker': ticker, 'datatype': datatype}
    result = requests.post(url, data=params).json()
    return result['return']['content']


def fill_multiple_data(datatypes, tickers):
    result = dict()
    for tk in tickers:
        try:
            result[tk] = fill_data(datatype=datatypes, ticker=tk)
        except Exception:
            result[tk] = 'Failed'
    return result

def sma_trade_recon(date, datatype, tickers, trade_record_path='Trade_Recon_Trade/'):
    import os
    import collections
    df = download_multiple_data(datatype, tickers)
    tmp = dict()
    files = os.listdir(trade_record_path)
    trades = {'BUY': collections.defaultdict(lambda: list()), 'SELL': collections.defaultdict(lambda: list())}
    dfs = {'BUY': collections.defaultdict(lambda: dict()), 'SELL': collections.defaultdict(lambda: dict())}
    for file in files:
        splits = file.split('_')
        ticker = splits[-1][:-5]
        year = splits[0]
        minute = int(splits[2]) + 1
        hour = int(splits[1])
        if minute == 60:
            hour = hour + 1
            minute = 0
        dt = datetime.datetime.strptime(year, "%Y%m%d") + datetime.timedelta(hours=hour, minutes=minute)
        trades[splits[-2]][ticker].append(dt)
        dfs[splits[-2]][ticker][pd.to_datetime(dt)] = pd.read_excel(trade_record_path+file)

    for tk in pd.unique([x[0] for x in df.index]):
        tk_df = df.loc[tk]
        tk_df = tk_df.loc[tk_df.datetime >= datetime.datetime.strptime(date, '%Y-%m-%d')]
        tk_df['SMA16'] = talib.SMA(tk_df['close'], timeperiod=10)
        tk_df['SMA32'] = talib.SMA(tk_df['close'], timeperiod=20)
        tk_df['buy_signal'] = (tk_df['SMA16'].shift(1) <= tk_df['SMA32'].shift(1)) * (tk_df['SMA16'] > tk_df['SMA32'])
        tk_df['sell_signal'] = -1 * (tk_df['SMA16'].shift(1) >= tk_df['SMA32'].shift(1)) * (
                    tk_df['SMA16'] < tk_df['SMA32'])
        tk_df['signal'] = tk_df['buy_signal'] + tk_df['sell_signal']
        tk_df['signal'] = tk_df['signal'].astype(int)
        for id, row in tk_df.loc[tk_df['signal'] != 0].iterrows():
            if row['signal'] == 1:
                if row['datetime'] not in trades['BUY'][tk]:
                    print(f'Missing Trade : {row["ticker"]} BUY at {row["datetime"]}')

            elif row['signal'] == -1:
                if row['datetime'] not in trades['SELL'][tk]:
                    print(f'Missing Trade : {row["ticker"]} SELL at {row["datetime"]}')
        tmp[tk] = tk_df
    return tmp

if __name__ == '__main__':
    tickers = ['HK.00700', 'HK.00388', 'HK.02318', 'HK.02020', 'HK.01299', 'HK.02018', 'HK.00291', 'HK.02382']
    tickers_2 = ['HK.00700', 'HK.54544554', 'HK.09988', 'HK.09999', 'HK.02318']
    # dfs = downlaod_multiple_daily(tickers)
    # print(fill_multiple_data('K_3M', tickers_2))
    result = sma_trade_recon('2020-08-06', 'K_3M', tickers_2)