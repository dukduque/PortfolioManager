# -*- coding: utf-8 -*-
import pandas as pd
from source import database_handler as dbh
data = dbh.load_database('close_2019-06-29.pkl')
data_r = dbh.quotien_diff(data)
data_bool = data_r<0.01
sum_rows = data_bool.sum(0)
sum_rows = sum_rows[sum_rows>=1]
bad_stocks = set(sum_rows.index)
total_out = sum_rows.sum()
total_stocks = (sum_rows>1).sum()
counter = 0 
for c in data_r.colums:
    counter
    
    




import yfinance as yf #Works awsome!
import pandas as pd

#Make the ticker a variable
drv = yf.Ticker('BHF')    
#Get all the available info for that ticker
info = drv.info  
#get all the events information (splits and dividends) for that ticker
events = drv.actions
#Get all price information for that ticker
#If we use auto_adjust = false we get prices adjusted only for splits. If true, we get prices adjusted for both splits and dividends
#Actions = true indica que nos de splits y dividendos
hist = drv.history(period="max",auto_adjust=False,actions=True) 
#We delete the columns related to Open, High, and low
hist = hist.drop(['Open','High','Low'],axis=1)
#We print the results
print(info)
print(events)
print(hist)
#---------------------



