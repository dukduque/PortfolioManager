'''
Various scripts to view and manage an account.
'''
from pathlib import Path
import sys
path = Path(__file__)
print(str(path.parent.parent))
sys.path.append(str(path.parent.parent))

from source.resources import Portfolio, Account, Order, load_account, generate_orders, save_account
from source.resources import OPERATION_BUY, OPERATION_SELL, build_account_history
import datetime as dt
from source import database_handler as dbh
from source.database_handler import DataManager
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


def read_account(account_name):
    account = load_account(account_name)
    print(account)


def add_transactions(account_name):
    account = load_account(account_name)
    account.deposit(dt.datetime(2021, 1, 12, 10, 0), 1000)
    account.update_account(dt.datetime(2021, 1, 15, 14, 00), [
        Order('AMD', 3, 89.14, OPERATION_BUY),
        Order('WMT', 3, 144.64, OPERATION_BUY),
        Order('MSFT', 1, 213.79, OPERATION_BUY),
        Order('BOTZ', 3, 34.64, OPERATION_BUY),
    ])
    return account


def benchmark(account_name):
    '''
        Benchmarks an account against the SP500
    '''
    account = load_account(account_name)
    
    file_name = 'close.pkl'
    data_manager = DataManager(db_file=file_name)
    sp500_stocks = dbh.get_sp500_tickers()
    
    sp500_portfolios = {}
    ini_portfolio = Portfolio.create_empty()
    SPY_prices = data_manager.get_prices("SPY")
    for op_date, op_value in account.operations_history():
        pandas_date = pd.Timestamp(year=op_date.year, month=op_date.month, day=op_date.day)
        date_ix = np.where(SPY_prices.index == pandas_date)[0][0]
        price_on_date = SPY_prices.iloc[date_ix]
        portfolio_on_date = Portfolio.create_from_vectors(assets=['SPY'], qty=[op_value / price_on_date])
        sp500_portfolios[op_date] = portfolio_on_date + ini_portfolio
        ini_portfolio = sp500_portfolios[op_date]
    history_dates, sp_500_history_values = build_account_history(sp500_portfolios, data_manager)
    history_dates, history_values = build_account_history(account.portfolios, data_manager)
    
    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.plot(history_dates, history_values, color='blue')
    axes.plot(history_dates, sp_500_history_values, color='red')
    plt.show()


def piechart(account_name):
    file_name = 'close.pkl'
    data_manager = DataManager(db_file=file_name)
    account = load_account(account_name)
    portfolio = account.portfolio
    assets = portfolio.assets
    price = data_manager.get_prices(assets).iloc[-1]  # Last row is the current price
    
    position = np.array([portfolio.get_position(a) * price[a] for a in assets])
    portfolio_value = position.sum()
    position /= portfolio_value
    
    fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
    axes.pie(position, labels=assets, autopct='%1.1f%%')
    axes.axis('equal')
    plt.show()
    
    # Pie by sectors
    sectors = defaultdict(float)
    for a in assets:
        total_position = portfolio.get_position(a) * price[a]
        marginal_position = total_position / portfolio_value
        sector = data_manager.get_metadata(a)['sector']
        print(f'{a} {sector}  {total_position:.2f} {marginal_position:.3f}')
        sectors[sector] += marginal_position
    
    fig, axes = plt.subplots(ncols=1, figsize=(5, 5))
    axes.pie(sectors.values(), labels=sectors.keys(), autopct='%1.1f%%')
    axes.axis('equal')
    plt.show()


if __name__ == "__main__":
    acc_name = "Daniel Duque"
    read_account(acc_name)
    # new_account = add_transactions(acc_name)
    # save_account(new_account)
    benchmark(acc_name)
    piechart(acc_name)
