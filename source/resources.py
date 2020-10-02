"""
This module contains various data structurs for data manipulation, optimization, and backtesting
"""
import os
import sys
import datetime as dt
import pandas as pd
import numpy as np
import copy
import pickle
from pathlib import Path

project_path = Path(__file__).parent.parent
accounts_path = project_path / 'accounts/'


class Portfolio:
    def __init__(self):
        self._data = {}
    
    def get_position(self, asset):
        if asset in self._data:
            return self._data[asset]["qty"]
        else:
            return None
    
    @classmethod
    def create_from_vectors(cls, assets, qty):
        p = cls()
        for (a, q) in zip(assets, qty):
            p._data[a] = {"qty": q}
        
        return p
    
    @classmethod
    def create_from_transaction(cls, portfolio, orders):
        p = copy.deepcopy(portfolio)
        for order in orders:
            if order.operation_type == OPERATION_SELL:
                p._data[order.ticker]['qty'] -= order.qty
                if p._data[order.ticker]['qty'] < 0:
                    return None
                if p._data[order.ticker]['qty'] == 0:
                    del p._data[orders.ticker]
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


OPERATION_SELL = 'sell'
OPERATION_BUY = 'buy'
OPERATION_TYPES = [OPERATION_SELL, OPERATION_BUY]


class Order:
    def __init__(self, ticker, qty, price, operation_type):
        self.ticker = ticker
        self.qty = qty
        self.price = price
        self.operation_type = operation_type


class Account:
    def __init__(self, holder_name, opening_date):
        self.holder = holder_name
        self.cash_flow = pd.DataFrame(columns=["datetime", "amount", "type"])
        self.transactions = pd.DataFrame(columns=["ticker", "datetime", "operation", "qty", "price"])
        self.portfolios = {opening_date: Portfolio.create_empty()}
        self.last_transaction = opening_date
        self.cash_onhand = 0
    
    def deposit(self, deposit_date, amount):
        self.cash_flow.append({
            "datetime": deposit_date,
            "amount": amount,
            "type": "deposit",
        }, ignore_index=True)
        self.cash_onhand = self.cash_onhand + amount
    
    def withdraw(self, withdraw_date, amount):
        '''
        Withdraw monay from cash onhand.
        '''
        if (self.cash_onhand < amount):
            return False
        self.cash_flow.append({
            "datetime": withdraw_date,
            "amount": amount,
            "type": "withdraw",
        }, ignore_index=True)
        self.cash_onhand = self.cash_onhand - amount
        return False
    
    def update_account(self, transaction_date, orders):
        '''
        Updates the account with orders that were succesfull and returns
        True if the update was successful.
        Args:
        transaction_date (datetime): date of the transaction.
        orders (list of Order): list of excuted orders.
        '''
        assert self.last_transaction <= transaction_date
        current_portfolio = self.portfolios[self.last_transaction]
        new_portfolio = Portfolio.create_from_transaction(current_portfolio, orders)
        if new_portfolio:
            self.last_transaction = transaction_date
            self.portfolios[transaction_date] = new_portfolio
            for order in orders:
                self.cash_flow.append(
                    {
                        "ticker": order.ticker,
                        "datetime": transaction_date,
                        "operation": order.operation_type,
                        "qty": order.qty,
                        "price": order.price,
                    },
                    ignore_index=True)
            return True
        return False
    
    def build_porfolio_snapshots(self, start=None, end=None, time_resolution="days"):
        '''
            Constructs the history of the account from the the start date given
            as a paramter.
            
            Args:
            start (datetime): start date from which the history will be computed
                If None, the histoty is computed form the account's opning date.
            end (datetime): end date from of the history. If None, the history is
                compute until dt.today().
            time_resolution (str): granularity of the history ("weeks", "days", "hours"),
                default is "days"
            
            Returns:
            history (dict of datetime-porfolios): a dictionary with all the porfolios that
                the account has 
        '''
        pass


def save_account(account):
    backup_name = str(dt.datetime.now()).replace(".", "").replace(":", "").replace(" ", "").replace("-", "") + ".acc"
    path_to_account = accounts_path / account.holder
    if not path_to_account.exists():
        path_to_account.mkdir()
    index_file = path_to_account / 'index.txt'
    with open(index_file, "a") as writer:
        writer.write(backup_name)
    backup_file = path_to_account / backup_name
    pickle.dump(account, backup_file.open("wb"), pickle.HIGHEST_PROTOCOL)


def load_account(account_name):
    path_to_account = accounts_path / account_name
    if not path_to_account.exists():
        return None
    acc_file = None
    with open(path_to_account / "index.txt") as index_file:
        hist = index_file.read()
        hist_list = hist.split(".acc")
        hist_list.reverse()
        for acc_backup in hist_list:
            backup_name = acc_backup + ".acc"
            try:
                backup_file = path_to_account / backup_name
                account = pickle.load(backup_file.open("rb"))
                return account
            except Exception as identifier:
                print(identifier)
                print("WARNING: account backup {backup_name} not loaded.")


if __name__ == '__main__':
    dd_account = Account("Daniel Duque", dt.datetime(2020, 9, 14, 9, 30))
    dd_account.deposit(dt.datetime(2020, 9, 14, 9, 30), 1000)
    dd_account.update_account(dt.datetime(2020, 9, 14, 9, 35), [
        Order('HRL', 1, 49.95, OPERATION_BUY),
        Order('NEM', 2, 66.54, OPERATION_BUY),
        Order('SEE', 1, 39.49, OPERATION_BUY),
        Order('MRO', 2, 4.47, OPERATION_BUY),
        Order('WMT', 2, 136.14, OPERATION_BUY),
        Order('CTXS', 1, 134.02, OPERATION_BUY),
        Order('VS', 1, 60.02, OPERATION_BUY),
        Order('KR', 3, 33.68, OPERATION_BUY),
        Order('HLT', 1, 88.18, OPERATION_BUY),
        Order('SJM', 1, 113.49, OPERATION_BUY),
    ])
    print(dd_account.cash_onhand)
    print(dd_account.portfolios)
    ##save_account(dd_account)
