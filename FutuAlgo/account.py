import datetime
import pandas as pd


class Account:
    """ Class that handle positions, cash and trades """
    def __init__(self, logger, initial_capital: float, txn_cost: float,):
        self._logger = logger

        self._pending_orders = dict()
        self._records = pd.DataFrame(columns=['PV', 'EV', 'Cash'])
        self._completed_orders = pd.DataFrame(columns=['order_id'])
        self._positions = pd.DataFrame(columns=['price', 'quantity', 'market_value'])
        self._slippage = pd.DataFrame(columns=['exp_price', 'dealt_price', 'dealt_qty', 'total_slippage'])

        self._txn_cost = txn_cost
        self._total_txn_cost = 0
        self._initial_capital = initial_capital
        self._current_cash = initial_capital

    def log(self, overwrite_date=None):
        ev = sum(self._positions['market_value'])
        pv = ev + self._current_cash
        d = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') if overwrite_date is None else overwrite_date
        self._records.loc[d] = [pv, ev, self._current_cash]

    def add_new_position(self, ticker):
        self._positions.loc[ticker] = [0.0, 0.0, 0.0]

    def update_positions(self, df):
        """ Update position and cash on receiving updates from broker """
        trd_side = 1 if df['trd_side'].iloc[0].upper() in ('BUY', 'BUY_BACK') else -1
        dealt_qty = df['dealt_qty'].iloc[0] * trd_side
        avg_price = df['dealt_avg_price'].iloc[0]
        order_id = df['order_id'].iloc[0]
        ticker = df['ticker'].iloc[0]
        order_status = df['order_status'].iloc[0]

        in_pending = False
        in_completed = False
        if order_id in self._pending_orders.keys():
            in_pending = True
            last_order_update_df = self._pending_orders[order_id]

            last_qty = last_order_update_df['dealt_qty'].iloc[0] * trd_side
            last_avg_price = last_order_update_df['dealt_avg_price'].iloc[0]

            cash_change = -(dealt_qty * avg_price - last_qty * last_avg_price)
            qty_change = dealt_qty - last_qty

        else:
            if order_id not in self._completed_orders['order_id']:
                cash_change = - dealt_qty * avg_price
                qty_change = dealt_qty
            else:
                in_completed = True
                cash_change = 0
                qty_change = 0

        if order_status in ('SUBMIT_FAILED', 'FILLED_ALL', 'CANCELLED_PART', 'CANCELLED_ALL', 'FAILED', 'DELETED'):
            if not in_completed:
                self._completed_orders = self._completed_orders.append(df)

                if order_status in ('FILLED_ALL', 'CANCELLED_PART'):
                    # update slippage
                    self._slippage.loc[order_id] = [0, 0, 0, 0]
                    exp_price = self._positions.loc[ticker]['price'] if df['price'].iloc[0] == 0.0 else df['price'].iloc[
                        0]
                    self._slippage.loc[order_id] = [exp_price, avg_price, dealt_qty, (avg_price - exp_price) * dealt_qty]

                    # Txn cost
                    self._total_txn_cost += self._txn_cost
                    cash_change -= self._txn_cost

                if in_pending:
                    del self._pending_orders[order_id]

        else:
            self._pending_orders[order_id] = df

        # update positions and snapshot
        latest_price = self._positions.loc[ticker]['price']
        existing_qty = self._positions.loc[ticker]['quantity']

        new_qty = existing_qty + qty_change
        self._positions.loc[ticker] = [latest_price, new_qty, new_qty * latest_price]
        self._current_cash += cash_change

    def update_prices(self, datatype, df):
        """ Update price for valuation of positions """
        if 'K_' in datatype:
            ticker = df['ticker'].iloc[0]
            qty = self._positions.loc[ticker]['quantity']
            latest_price = df['close'].iloc[0]
            self._positions.loc[ticker] = [latest_price, qty, qty * latest_price]

        elif datatype == 'QUOTE':
            ticker = df['ticker'].iloc[0]
            qty = self._positions.loc[ticker]['quantity']
            latest_price = df['quote'].iloc[0]
            self._positions.loc[ticker] = [latest_price, qty, qty * latest_price]

    def pre_trade_check(self, ticker, quantity, trade_side, price, lot_size):
        trade_sign = -1 if trade_side == 'BUY' else 1
        price = self._positions.loc[ticker]['price'] if price == 0.0 else price
        exp_cash_change = price * quantity * trade_sign

        if quantity <= 0:
            return 0, f'Quantity cannot be <= 0! '

        if self._current_cash + exp_cash_change < 0:
            return 0, f'Not enough cash, current cash:{self._current_cash} , required cash:{-exp_cash_change}'

        if quantity % lot_size != 0:
            return 0, f'Lot size is invalid, should be multiple of {lot_size} but got {quantity}'

        return 1, 'Risk check passed'

    # ------------------------------------------------ [ Get infos ] ------------------------------------------
    def get_current_qty(self, ticker):
        return self._positions.loc[ticker]['quantity']

    def get_latest_price(self, ticker):
        return self._positions.loc[ticker]['price']

    def calc_max_buy_qty(self, ticker, lot_size, cash=None, adjust_limit=1.03):
        cash = self._current_cash if cash is None else cash
        one_hand_size = lot_size * self.get_latest_price(ticker) * adjust_limit
        if cash >= one_hand_size:
            max_qty_by_cash = int((cash - cash % one_hand_size) / one_hand_size) * lot_size
            return max_qty_by_cash
        else:
            return 0

    @property
    def pv(self):
        return self.mv + self.cash

    @property
    def mv(self):
        return sum(self._positions['market_value'])

    @property
    def cash(self):
        return self._current_cash

    @property
    def completed_orders(self):
        return self._completed_orders.copy()

    @property
    def pending_orders(self):
        return self._pending_orders.copy()

    @property
    def n_trades(self):
        return self._completed_orders.shape[0]

    @property
    def init_capital(self):
        return self._initial_capital

    @property
    def positions(self):
        return self._positions.copy()

    @property
    def slippage(self):
        return self._slippage.copy()

    @property
    def total_txn_cost(self):
        return self._total_txn_cost

    @property
    def records(self):
        return self._records.copy()
