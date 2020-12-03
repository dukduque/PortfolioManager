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
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


def read_account(account_name):
    account = load_account(account_name)
    print(account)


def add_transactions(account_name):
    account = load_account(account_name)
    account.deposit(dt.datetime(2020, 11, 30, 9, 0), 3.636 * 1768.88)
    account.update_account(dt.datetime(2020, 11, 30, 15, 0), [
        Order('GOOG', 3.636, 1768.88, OPERATION_BUY),
    ])
    return account


def benchmark(account_name):
    '''
        Benchmarks an account against the SP500
    '''
    file_name = 'close.pkl'
    start_date = dt.datetime(2018, 11, 14)
    end_date_train = dt.datetime(2020, 11, 14)
    end_date_test = dt.datetime(2021, 9, 4)
    outlier_return = 10
    sp500_stocks = dbh.get_sp500_tickers()
    db, _ = dbh.get_returns(data_file=file_name,
                            start_date=start_date,
                            stocks=list(sp500_stocks.keys()),
                            outlier_return=outlier_return)
    price = db.iloc[-1]  # Last row is the current price
    account = load_account(account_name)
    
    sp500_portfolios = {}
    sp500_value = sum(sp500_stocks[s]['market_cap'] for s in db.columns)
    allocation = np.array([sp500_stocks[s]['market_cap'] for s in db.columns]) / sp500_value
    ini_portfolio = Portfolio.create_empty()
    for op_date, op_value in account.operations_history():
        pandas_date = pd.Timestamp(year=op_date.year, month=op_date.month, day=op_date.day)
        date_ix = np.where(db.index == pandas_date)[0][0]
        price_on_date = 0.2 * db.iloc[date_ix] + 0.8 * db.iloc[date_ix - 1]
        portfolio_on_date = Portfolio.create_from_vectors(assets=price_on_date.index,
                                                          qty=allocation * op_value / price_on_date)
        sp500_portfolios[op_date] = portfolio_on_date + ini_portfolio
        ini_portfolio = sp500_portfolios[op_date]
    history_dates, sp_500_history_values = build_account_history(sp500_portfolios, db)
    history_dates, history_values = build_account_history(account.portfolios, db)
    
    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.plot(history_dates, history_values, color='blue')
    axes.plot(history_dates, sp_500_history_values, color='red')
    plt.show()


def piechart(account_name):
    file_name = 'close.pkl'
    sp500_stocks = dbh.get_sp500_tickers()
    db, _ = dbh.get_returns(data_file=file_name,
                            start_date=dt.datetime.today() - dt.timedelta(5),
                            stocks=list(sp500_stocks.keys()),
                            outlier_return=10)
    price = db.iloc[-1]  # Last row is the current price
    account = load_account(account_name)
    portfolio = account.portfolio
    assets = portfolio.assets
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
        print(f'{a} {sp500_stocks[a]["sector"]}  {total_position:.2f} {marginal_position:.3f}')
        sectors[sp500_stocks[a]['sector']] += marginal_position
    
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
