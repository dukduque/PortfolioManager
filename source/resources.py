"""
This module contains various data structurs for data manipulation, optimization, and backtesting
"""

import datetime as dt
import pandas as pd
import numpy as np


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
        p = cls()
        return p
    
    @classmethod
    def create_empty(cls):
        p = cls()


class Order:
    def __init__(self, ticker, qty, price, op_type):
        self.ticker = ticker
        self.qty = qty
        self.price = price
        self.op_type = op_type


class Account:
    def __init__(self, holder_name, opening_date):
        self.holder = holder_name
        self.deposits = pd.DataFrame(columns=["datetime", "amount"]),
        self.transactions = pd.DataFrame(columns=["ticker", "datetime", "operation", "qty", "price"])
        self.portfolios = {opening_date: Portfolio.create_empty()}
        self.last_transaction = opening_date
    
    def deposit(self, deposit_date, amount):
        self.deposits.append({
            "datetime": deposit_date,
            "amount": amount,
        }, ignore_index=True)
    
    def withdraw(self, withdraw_date, amount):
        return False
    
    def update_account(self, transaction_date, orders):
        assert self.last_transaction <= transaction_date
        current_portfolio = self.portfolios[self.last_transaction]
        new_portfolio = Portfolio.create_from_transaction(current_portfolio, orders)
        
        self.last_transaction = transaction_date
        self.portfolios = []
    
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