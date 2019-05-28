#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:36:26 2019

@author: dduque
"""
'''
Setup paths
'''
import os 
import sys
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.append(parent_path)

from source.opt_tools import cvar_model
import source.database_handler as dbh
import backtest as bt
import pandas as pd
import datetime as dt


my_date = dt.datetime(2013,1,1)
sp500 = dbh.save_sp500_tickers()
db, data = dbh.get_returns(data_file = 'close_2019-05-26.pkl', 
                           start_date= my_date,
                           stocks=sp500)#, stocks=['ABC','MSFT', 'AMZN', 'GOOGL', 'GE', 'F', 'MMM', 'ATVI'])

#out_dro_model = markovitz_dro_wasserstein(data, 0.001,1.001)

price = db.iloc[-1] #Last row is the current price
cvar_mod = cvar_model(data, price, budget=2000, fractional=False)

portfolios = []
portfolio_stats = []
for cvar_beta in [0.5,0.9,0.99]:
    cvar_sol1, cvar_stats1 = cvar_mod.change_cvar_params(cvar_beta=cvar_beta)
    portfolios.append(cvar_sol1[cvar_sol1.stock>0])
    portfolio_stats.append(cvar_stats1)

portfolio_paths = []
for p in portfolios:
    portfolio_paths.append(bt.run_backtest(p, db, dt.datetime(2013,1,1), dt.datetime(2019,5,1)))

bt.plot_backtests(portfolio_paths)



#out_file = dbh.path_to_output + '/cvar_sol.csv'
#cvar_sol.to_csv(out_file)
#print(cvar_sol[cvar_sol.position>1000])
#print(cvar_sol1[cvar_sol1.stock>=1])
#print(cvar_stats)
#solution = pd.DataFrame({'allocation':out_dro_model[0]},index=db.columns)
