'''
Various scripts to view and manage an account.
'''
import sys

from resources import Portfolio, Account, Order, load_account, \
    generate_orders, save_account
from resources import OPERATION_BUY, OPERATION_SELL, \
    build_account_history
import datetime as dt
import database_handler as dbh
from database_handler import DataManager
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


def read_account(account_name):
    account = load_account(account_name)
    print(account)


def benchmark(account_name, benchmark_symbol='SPY'):
    '''
        Benchmarks an account against `benchmark_symbol`.
    '''
    account = load_account(account_name)
    
    file_name = 'close.pkl'
    data_manager = DataManager(db_file=file_name)
    sp500_stocks = dbh.get_sp500_tickers()
    
    sp500_portfolios = {}
    ini_portfolio = Portfolio.create_empty()
    SPY_prices = data_manager.get_prices(benchmark_symbol)
    net_transactions = {}
    balance = 0
    for op_date, op_value in account.operations_history():
        balance += op_value
        pandas_date = pd.Timestamp(year=op_date.year,
                                   month=op_date.month,
                                   day=op_date.day)
        date_ix = np.where(SPY_prices.index == pandas_date)[0][0]
        price_on_date = SPY_prices.iloc[date_ix]
        portfolio_on_date = Portfolio.create_from_vectors(
            assets=[benchmark_symbol], qty=[op_value / price_on_date])
        sp500_portfolios[op_date] = portfolio_on_date + ini_portfolio
        ini_portfolio = sp500_portfolios[op_date]
        net_transactions[pandas_date] = balance
    
    history_dates, sp_500_history_values = build_account_history(
        sp500_portfolios, data_manager)
    history_dates, history_values = build_account_history(
        account.portfolios, data_manager)
    #print(history_dates)
    #print(net_transactions.keys())
    
    transaction_dates = list(net_transactions.keys())
    transaction_dates.sort()
    cummulative_transactions = [0] * len(history_dates)
    last_date_ix = 0
    for d_ix, d in enumerate(history_dates):
        for d_transaction in transaction_dates:
            if d > d_transaction:
                cummulative_transactions[d_ix] = net_transactions[d_transaction]
    today = dt.datetime.today()
    cummulative_transactions = [net_transactions[d] for d in transaction_dates]
    transaction_dates.append(
        pd.Timestamp(year=today.year, month=today.month, day=today.day))
    cummulative_transactions.append(cummulative_transactions[-1])
    
    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.step(transaction_dates,
              cummulative_transactions,
              where='post',
              color='lightblue',
              alpha=0.7)
    axes.plot(history_dates, history_values, color='blue')
    axes.plot(history_dates, sp_500_history_values, color='red')
    
    plt.show()


def piechart(account_name):
    file_name = 'close.pkl'
    data_manager = DataManager(db_file=file_name)
    account = load_account(account_name)
    portfolio = account.portfolio
    assets = list(portfolio.assets)
    price = data_manager.get_prices(assets).iloc[
        -1]  # Last row is the current price
    if price.isnull().sum() > 0:
        price = data_manager.get_prices(assets).iloc[-2]
    assets.sort(key=lambda x: portfolio.get_position(x) * price[x],
                reverse=True)
    position = np.array([portfolio.get_position(a) * price[a] for a in assets])
    portfolio_value = position.sum()
    position /= portfolio_value
    
    fig, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.pie(position,
             labels=assets,
             autopct='%1.1f%%',
             pctdistance=0.9,
             labeldistance=1.05,
             startangle=45)
    axes.axis('equal')
    plt.show()
    
    # Pie by sectors
    sectors = defaultdict(float)
    for a in assets:
        total_position = portfolio.get_position(a) * price[a]
        marginal_position = total_position / portfolio_value
        sector = data_manager.get_metadata(a)['sector']
        print(
            f'{a:5s} {sector:25s}  {total_position:7.2f} {marginal_position * 100:5.3f}%'
        )
        sectors[sector] += marginal_position
    print(f'Porfolio value: {portfolio_value:10.2f}')
    
    fig, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.pie(sectors.values(),
             labels=sectors.keys(),
             autopct='%1.1f%%',
             pctdistance=0.9,
             labeldistance=1.05,
             startangle=45)
    axes.axis('equal')
    plt.show()
