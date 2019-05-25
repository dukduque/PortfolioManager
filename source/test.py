#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:36:26 2019

@author: dduque
"""


from source.opt_tools import markovitz_dro_wasserstein, cvar_model
import source.database_handler as dbh
import pandas as pd
import datetime as dt

my_date = dt.datetime(2007,1,1)
db, data = dbh.get_returns(start_date= my_date)#, stocks=['ABC','MSFT', 'AMZN', 'GOOGL', 'GE', 'F', 'MMM', 'ATVI'])

out_dro_model = markovitz_dro_wasserstein(data, 0.001,1.001)

price = db.iloc[-1] #Last row is the current price
cvar_sol, cvar_stats = cvar_model(data, price, budget=100000, fractional=False)
out_file = dbh.path_to_output + '/cvar_sol.csv'
cvar_sol.to_csv(out_file)
print(cvar_sol[cvar_sol.position>1000])
print(cvar_sol[cvar_sol.stock>=1])
print(cvar_stats)
solution = pd.DataFrame({'allocation':out_dro_model[0]},index=db.columns)


