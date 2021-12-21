"""
This module contains various data structures for data manipulation,
optimization, and backtesting
"""
import datetime as dt
import pandas as pd
import numpy as np
import copy
import pickle
import math
import types
from pathlib import Path
from collections import defaultdict

project_path = Path(__file__).parent.parent
accounts_path = project_path / 'accounts/'


class Portfolio:
    '''
    '''
    def __init__(self):
        self._data = {}
    
    def get_position(self, asset):
        if asset in self._data:
            return self._data[asset]["qty"]
        else:
            return 0
    
    def position_is_fractional(self, asset):
        position = self.get_position(asset)
        if position - int(position) > 0:
            return True
        else:
            return False
    
    @property
    def assets(self):
        return self._data.keys()
    
    @classmethod
    def create_from_vectors(cls, assets, qty):
        p = cls()
        for (a, q) in zip(assets, qty):
            if q < 0:
                return None
            p._data[a] = {"qty": q}
        return p
    
    @classmethod
    def create_from_transaction(cls, portfolio, orders):
        p = copy.deepcopy(portfolio)
        for order in orders:
            if order.operation_type == OPERATION_SELL:
                if (order.ticker not in p._data and order.qty > 0):
                    return None
                p._data[order.ticker]['qty'] -= order.qty
                if p._data[order.ticker]['qty'] < 0:
                    return None
                if p._data[order.ticker]['qty'] == 0:
                    del p._data[order.ticker]
            if order.operation_type == OPERATION_BUY:
                if order.ticker in p._data:
                    p._data[order.ticker]['qty'] += order.qty
                else:
                    p._data[order.ticker] = {'qty': order.qty}
        return p
    
    @classmethod
    def create_empty(cls):
        p = cls()
        return p
    
    def __str__(self):
        p_string = ''
        for a in self._data:
            p_string += f'{a:8s}{self._data[a]["qty"]}\n'
        return p_string
    
    def __repr__(self):
        return self.__str__()
    
    def __add__(self, other_portfolio):
        if other_portfolio is None:
            return None
        all_assets = list(set(list(self.assets) + list(other_portfolio.assets)))
        qtys = []
        for a in all_assets:
            if self.get_position(a) < 0:
                return None
            if other_portfolio.get_position(a) < 0:
                return None
            q = self.get_position(a) + other_portfolio.get_position(a)
            qtys.append(q)
        return Portfolio.create_from_vectors(all_assets, qtys)


OPERATION_SELL = 'Sell'
OPERATION_BUY = 'Buy'
OPERATION_TYPES = [OPERATION_SELL, OPERATION_BUY]


class Order:
    def __init__(self, ticker, qty, price, operation_type):
        self.ticker = ticker
        self.qty = qty
        self.price = price
        self.operation_type = operation_type
    
    def __str__(self):
        return f"{self.operation_type} {self.ticker} : {self.qty}" + \
            f" : {self.price:.4f}"
    
    def __repr__(self):
        return self.__str__()


class Account:
    def __init__(self, holder_name, opening_date):
        self.holder = holder_name
        self.cash_flow = pd.DataFrame(columns=["datetime", "amount", "type"])
        self.transactions = pd.DataFrame(
            columns=["ticker", "datetime", "operation", "qty", "price"])
        self.portfolios = {opening_date: Portfolio.create_empty()}
        self.last_transaction = opening_date
        self.cash_onhand = 0
    
    @property
    def portfolio(self):
        '''
        Most recent portfolio of the account.
        '''
        return self.portfolios[self.last_transaction]
    
    def deposit(self, deposit_date, amount):
        self.cash_flow = self.cash_flow.append(
            {
                "datetime": deposit_date,
                "amount": amount,
                "type": "deposit",
            },
            ignore_index=True)
        self.cash_onhand = self.cash_onhand + amount
        return True
    
    def withdraw(self, withdraw_date, amount):
        '''
        Withdraw money from cash onhand.
        '''
        if (self.cash_onhand < amount):
            return False
        self.cash_flow = self.cash_flow.append(
            {
                "datetime": withdraw_date,
                "amount": amount,
                "type": "withdraw",
            },
            ignore_index=True)
        self.cash_onhand = self.cash_onhand - amount
        return True
    
    def update_account(self, transaction_date, orders):
        '''
        Updates the account with orders that were successful and returns
        True if the update was successful.
        Args:
        transaction_date (datetime): date of the transaction.
        orders (list of Order): list of executed orders.
        '''
        assert self.last_transaction <= transaction_date
        current_portfolio = self.portfolios[self.last_transaction]
        new_portfolio = Portfolio.create_from_transaction(
            current_portfolio, orders)
        if new_portfolio:
            self.last_transaction = transaction_date
            self.portfolios[transaction_date] = new_portfolio
            for order in orders:
                self.transactions = self.transactions.append(
                    {
                        "ticker": order.ticker,
                        "datetime": transaction_date,
                        "operation": order.operation_type,
                        "qty": order.qty,
                        "price": order.price,
                    },
                    ignore_index=True)
                if order.operation_type == OPERATION_BUY:
                    self.cash_onhand -= order.qty * order.price
                elif order.operation_type == OPERATION_SELL:
                    self.cash_onhand += order.qty * order.price
            return True
        return False
    
    def operations_history(self):
        history = []
        transactions_copy = self.transactions.copy()
        transactions_copy[
            'value'] = transactions_copy['price'] * transactions_copy['qty']
        operations = transactions_copy.groupby(['datetime', 'operation'
                                                ])['value'].agg('sum')
        for ix, value in zip(operations.index, operations):
            op_datetime, op_type = ix
            signed_value = value if op_type == OPERATION_BUY else -value
            history.append((op_datetime, signed_value))
        return history
    
    def __str__(self):
        s = f"Holder: {self.holder} - cash onhand: {self.cash_onhand:.2f}\n"
        s += f"Last transaction: {self.last_transaction} \n"
        s += f"Portfolio:\n{self.portfolio}"
        return s


def set_account_path(new_path_to_account):
    global accounts_path
    accounts_path = new_path_to_account


def create_new_account(account_name, opening_date=dt.datetime.now):
    '''
    Creates a new account instance an saves it at `accounts_path/account_name`
    '''
    account = Account(account_name, opening_date)
    save_account(account)
    return account


def save_account(account):
    backup_name = str(dt.datetime.now()).replace(".", "").replace(
        ":", "").replace(" ", "").replace("-", "") + ".acc"
    path_to_account = accounts_path / account.holder
    if not path_to_account.exists():
        path_to_account.mkdir()
    
    index_file = path_to_account / 'index.txt'
    with open(index_file, "a") as writer:
        writer.write(backup_name)
    backup_file = path_to_account / backup_name
    pickle.dump(account, backup_file.open("wb"), pickle.HIGHEST_PROTOCOL)
    
    # Save classes for future migration
    path_portfolio_class = path_to_account / 'PortfolioClass'
    pickle.dump(Portfolio, path_portfolio_class.open("wb"),
                pickle.HIGHEST_PROTOCOL)
    path_account_class = path_to_account / 'AccountClass'
    pickle.dump(Account, path_account_class.open("wb"), pickle.HIGHEST_PROTOCOL)


def load_account(account_name):
    path_to_account = accounts_path / account_name
    path_portfolio_class = path_to_account / 'PortfolioClass'
    path_account_class = path_to_account / 'AccountClass'
    if not path_to_account.exists():
        return None
    
    with open(path_to_account / "index.txt") as index_file:
        hist = index_file.read()
        hist_list = hist.split(".acc")
        hist_list.reverse()
        for acc_backup in hist_list:
            if len(acc_backup) > 0:
                backup_name = acc_backup + ".acc"
                try:
                    backup_file = path_to_account / backup_name
                    account = pickle.load(backup_file.open("rb"))
                    print(
                        f"INFO: account backup {backup_name} loaded successfully."
                    )
                    return account
                except Exception as identifier:
                    print(identifier)
                    print(f"WARNING: account backup {backup_name} not loaded.")
    
    return None


def generate_orders(old_portfolio, new_portfolio, prices):
    '''
        Generates a list of orders given two portfolios and the prices used to
        optimize the new portfolio. Note that buy/sell prices for these orders
        might be (slightly) different since the orders are placed manually
        after generating them.
    '''
    orders = []
    assets = set()
    assets.update(old_portfolio.assets)
    assets.update(new_portfolio.assets)
    for asset in assets:
        if asset not in prices:
            return None
        old_position = old_portfolio.get_position(asset)
        new_position = new_portfolio.get_position(asset)
        if old_position < new_position:  # Buy
            orders.append(
                Order(asset, new_position - old_position, prices[asset],
                      OPERATION_BUY))
        elif old_position > new_position:  # Sell
            orders.append(
                Order(asset, old_position - new_position, prices[asset],
                      OPERATION_SELL))
    orders.sort(key=lambda x: x.ticker)
    return orders


def build_account_history(portfolios, data_manager):
    '''
        Build a time series of the total value of the portfolios that
        an account has had since its opening until the last date in
        `prices_history`.

        Args:
            portfolios (dict of string-Portfolio): portfolios of the account
            at different transaction dates.
            data_manager (DataManager): instance of a data manager to get
            prices from.
    '''
    account_history = []
    history_dates = []
    portfolio_dates = list(portfolios.keys())
    portfolio_dates.sort()
    current_portfolio = None
    last_valid_price = defaultdict(float)
    for date_ix, date in enumerate(portfolio_dates):
        current_portfolio = portfolios[date]
        if len(current_portfolio.assets) == 0:
            continue
        assets_data = data_manager.get_prices(current_portfolio.assets)
        assert len(assets_data.columns) == len(current_portfolio.assets)
        start_date = dt.datetime(date.year, date.month, date.day)
        end_date = portfolio_dates[date_ix + 1] if date_ix + 1 <= len(
            portfolio_dates) - 1 else dt.datetime.today()
        end_date = dt.datetime(end_date.year, end_date.month, end_date.day)
        assets_data = assets_data[assets_data.index >= start_date]
        assets_data = assets_data[assets_data.index <= end_date]
        
        for d in assets_data.index:
            total_assets_d = 0
            prices_d = assets_data.loc[d]
            for asset in prices_d.index:
                asset_price_at_d = prices_d.loc[asset]
                if (not math.isnan(asset_price_at_d)):
                    last_valid_price[asset] = asset_price_at_d
                total_assets_d += last_valid_price[
                    asset] * current_portfolio.get_position(asset)
            account_history.append(total_assets_d)
            history_dates.append(d)
    
    return history_dates, account_history
