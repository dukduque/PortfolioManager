"""
Created on Thu May 23 22:36:26 2019
@author: dduque
"""
'''
Setup paths
'''
import sys
import os
import datetime as dt
import pandas as pd
import numpy as np

import backtest as bt
import database_handler as dbh
from opt_tools import cvar_model_pulp, cvar_model_ortools
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.append(parent_path)
file_name = 'close.pkl'
start_date = dt.datetime(2016, 6, 20)
sp500 = dbh.yf.Ticker("^GSPC")  # Ticker
sp500history = sp500.history(period='max', interval='1d')['Close']
sp500history = sp500history[sp500history.index >= start_date]
sp500_stocks = dbh.save_sp500_tickers()
sp500_stocks_tickers = list(sp500_stocks.keys())
db_all, _ = dbh.get_returns(data_file=file_name, start_date=start_date, stocks=sp500_stocks_tickers)

# Train set
end_date_train = dt.datetime(2019, 5, 24)
stock_universe = list(db_all.columns)
db, db_r = dbh.get_returns(data_file=file_name, start_date=start_date, end_date=end_date_train, stocks=stock_universe)
# ['ABC','MSFT', 'AMZN', 'GOOGL', 'GE', 'F', 'MMM', 'ATVI'])
data = np.array(db_r)
'''
Create model with default parameters
'''
ini_capital = 1_000
price = db.iloc[-1]  # Last row is the current price
print('Buy date: ', db.index.values[-1])
n_stocks = len(db_r.columns)
sp500_benchmark = pd.Series(data=np.ones(n_stocks) / n_stocks, index=db_r.columns)
# opt_model = cvar_model_pulp(data, price, budget=ini_capital, fractional=False)
opt_model = cvar_model_ortools(data, price, budget=ini_capital, fractional=False)
'''
Solve parametricly in \beta
'''
portfolios = []
portfolio_stats = []
portfolio_names = []
for cvar_beta in [0.5]:  # [0.3, 0.5, 0.7, 0.9, 0.99]:
    cvar_sol1, cvar_stats1 = opt_model.change_cvar_params(cvar_beta=cvar_beta)
    portfolios.append(cvar_sol1[cvar_sol1.qty > 0])
    portfolio_stats.append(cvar_stats1)
    portfolio_names.append(f'cvar_{cvar_beta}')

# SP500 porfolio
allocation = np.ones(n_stocks) / n_stocks
sp500_portfolio = pd.DataFrame({
    'price': price,
    'qty': allocation * ini_capital / price,
    'position': allocation * ini_capital,
    'allocation': allocation,
    'side': 'buy'
})
portfolios.append(sp500_portfolio)
stats_sp_500 = {'mean': opt_model.r_bar.dot(allocation), 'std': np.sqrt(allocation.dot(opt_model.cov.dot(allocation)))}
portfolio_stats.append(stats_sp_500)
portfolio_names.append(f'SP500')

for (p, ps) in zip(portfolios, portfolio_stats):
    p2 = p.copy()
    p2['name'] = [sp500_stocks[s]['name'] for s in p.index]
    p2['sector'] = [sp500_stocks[s]['sector'] for s in p.index]
    p2['subsector'] = [sp500_stocks[s]['subsector'] for s in p.index]
    print(p2, ps)
print(portfolio_names)
import pickle
out_portfolios = portfolios, portfolio_stats
pickle.dump(out_portfolios, open('./cvar_portfolio.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

# Back test
start_date_test = end_date_train
end_date_test = dt.datetime(2020, 9, 8)
db = db_all[(db_all.index >= start_date_test) & (db_all.index <= end_date_test)]
portfolio_paths = []
for p in portfolios[:]:
    portfolio_paths.append(bt.run_backtest(p, db, start_date_test, end_date_test, 20))

sp500history = np.array(sp500history[(sp500history.index >= start_date_test) & (sp500history.index <= end_date_test)])
factor = ini_capital / sp500history[0]
sp500history = factor * sp500history
# portfolio_paths.append([sp500history])
bt.plot_backtests(portfolio_paths, portfolio_names)
'''
# Gurobi implmentation
cvar_gurobi = cvar_model(data, price, budget=10000, fractional=False)
gurobi_portfolio, gurobi_stats = cvar_gurobi.optimize()

portfolios = []
portfolio_stats = []
for cvar_beta in [0.5,0.9,0.99]:
    cvar_sol1, cvar_stats1 = cvar_mod.change_cvar_params(cvar_beta=cvar_beta)
    portfolios.append(cvar_sol1[cvar_sol1.stock>0])
    portfolio_stats.append(cvar_stats1)

portfolio_paths = []
for p in portfolios[:2]:
    portfolio_paths.append(bt.run_backtest(p, db, dt.datetime(2013,1,1), dt.datetime(2019,5,1), 6))

bt.plot_backtests(portfolio_paths)

'''
