from FutuAlgo.algo import Algo
import collections


class CandlestickStrategy(Algo):
    def __init__(self, name: str, bars_window: int, benchmark: str = 'HSI'):
        super().__init__(name=name, benchmark=benchmark)
        self._last_update = collections.defaultdict(lambda: collections.defaultdict(lambda: None))
        self._bars_window = bars_window

    def determine_trigger(self, datatype, ticker, df):
        if 'K_' in datatype:
            datetime = df['datetime'].iloc[-1]
            last_df = self._last_update[ticker][datatype]
            trigger_strat = (last_df is not None) and (datetime != last_df['datetime'].iloc[-1])
            self._last_update[ticker][datatype] = df
            return trigger_strat, (
                datatype, ticker, self.get_data(datatype=datatype, ticker=ticker, n_rows=self._bars_window + 1)[:-1])
        else:
            return True, (datatype, ticker, df)
