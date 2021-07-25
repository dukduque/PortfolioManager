"""
Created on Thu May 23 22:36:26 2019
@author: dduque
"""
import sys
import os
import datetime as dt
import pandas as pd
import numpy as np
import backtest as bt
import database_handler as dbh
from source.database_handler import DataManager
from opt_tools import cvar_model_pulp, cvar_model_ortools
from resources import Portfolio, Account, load_account, generate_orders, save_account, build_account_history

# Setup paths
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.append(parent_path)

# Data prep
data_manager = DataManager('close.pkl', 'metadata.pkl')
start_date = dt.datetime(2019, 7, 13)  # Initial date for the training period
end_date = dt.datetime(2021, 7, 13)  # Final date for the training period
end_date_test = dt.datetime(2021, 9, 4)  # Final date for the backtesting period
ini_capital = 2000  # Initial capital available to be invested

# Training set
stock_universe = data_manager.securities
returns = data_manager.get_returns(start_date, end_date)
returns_array = np.array(returns)
current_price = data_manager.get_prices(returns.columns).loc[end_date]

# Create model with default parameters
account = load_account("Daniel Duque")
base_portfolio = account.portfolio
print(base_portfolio)
opt_model = cvar_model_ortools(returns_array,
                               current_price,
                               cvar_alpha=0.90,
                               current_portfolio=base_portfolio,
                               budget=ini_capital,
                               fractional=False,
                               portfolio_delta=0,
                               ignore=['GME', 'GOVT', 'BND', 'BLV', 'AR', 'RRC', 'SM', 'CDEV'])
'''
Solve parametricly in beta
'''
portfolios = []
portfolio_stats = []
portfolio_names = []
for cvar_beta in [0.95]:  # [i / 10 for i in range(1)]:
    cvar_sol1, cvar_stats1 = opt_model.change_cvar_params(cvar_beta=cvar_beta)
    portfolios.append(cvar_sol1[cvar_sol1.qty > 0])
    portfolio_stats.append(cvar_stats1)
    portfolio_names.append(f'cvar_{cvar_beta}')
    orders = generate_orders(base_portfolio, Portfolio.create_from_vectors(portfolios[-1].index, portfolios[-1].qty),
                             current_price)
    for o in orders:
        print(o)

for (p, ps) in zip(portfolios, portfolio_stats):
    p2 = p.copy()
    p2['name'] = [data_manager.get_metadata(s)['name'] for s in p.index]
    p2['sector'] = [data_manager.get_metadata(s)['sector'] for s in p.index]
    p2['subsector'] = [data_manager.get_metadata(s)['subsector'] for s in p.index]
    print(p2, '\n')
    print(ps)
print(portfolio_names)
import pickle
out_portfolios = portfolios, portfolio_stats, portfolio_names
pickle.dump(out_portfolios, open('./cvar_portfolio.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
