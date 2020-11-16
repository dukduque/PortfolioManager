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
from matplotlib import pyplot as plt


def read_account(account_name):
    account = load_account(account_name)
    print(account)


def add_transactions(account_name):
    account = load_account(account_name)
    account.deposit(dt.datetime(2020, 11, 13, 9, 55), 1000)
    account.update_account(dt.datetime(2020, 11, 13, 10, 1), [
        Order('VZ', 2, 60.46, OPERATION_BUY),
        Order('WMT', 3, 148.17, OPERATION_BUY),
        Order('CHRW', 2, 92.1, OPERATION_BUY),
        Order('HRL', 3, 51.89, OPERATION_BUY),
        Order('KR', 3, 31.91, OPERATION_BUY),
    ])
    print(account)
    
    # save_account(account_name)


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
    history_dates, sp_500_history_values = build_account_history(sp500_portfolios, db, None)
    history_dates, history_values = build_account_history(account.portfolios, db, sp_500_history_values)
    
    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.plot(history_dates, history_values, color='blue')
    axes.plot(history_dates, sp_500_history_values, color='red')
    plt.show()


if __name__ == "__main__":
    acc_name = "Daniel Duque"
    read_account(acc_name)
    # add_transactions(acc_name)
    benchmark(acc_name)
