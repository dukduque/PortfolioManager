#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:32:31 2019

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
path_to_data = os.path.abspath(os.path.join(parent_path,  'data'))

'''
Setup libraries
'''
import pandas as pd
import numpy as np
import pandas_datareader.robinhood as rh
import pandas_datareader.data as web
import datetime
import pickle


def load_database(DB_file_name):
    path_to_database = os.path.join(path_to_data,DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists == False:
        return None
    
    try:
        return pd.read_pickle(path_to_database)
    except Exception as e: 
        print(e)
        
def safe_database(BD,DB_file_name):
    path_to_database = os.path.join(path_to_data,DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists:
        #TODO: Make a copy of the existing file
        pass
    BD.to_pickle(path_to_database)



def create_database(stock_symbol='GOOGLE', start = None, end = None):
    db = web.DataReader(stock_symbol, 'yahoo', start=start, end=end)
    db = db.Close
    db = db.loc[~db.index.duplicated(keep='first')]
    db.rename(stock_symbol, inplace=True)
    return db
    

def add_stock(db, stock_symbol, start=None, end=None):
    ndb = create_database(stock_symbol, start, end)
    return pd.concat((db,ndb),axis=1,join='outer')
    

def quotien_diff(x):
    return np.divide(x[1:],x[:-1])
 

def get_returns(from_year):
    assert from_year >= 1970, 'Year should be from 1970'
    db = load_database('close_2019_05_24.pkl')
    db = db[db.index>'%i' %(from_year)]
    db = db.dropna(axis=1)
    db_r = db.apply(quotien_diff,axis=0) #compute returns
    
    r_data = np.array(db_r)
    return db, r_data

if __name__ == '__main__':

    symbols_df = web.get_nasdaq_symbols()
    symbols_df = symbols_df[symbols_df.ETF==False]
    sym_list = list(symbols_df.index)
    
    #db1 = create_database(sym_list[0], start=1900)
    for i in range(3354,len(sym_list)):
        try:
            db1 = add_stock(db1,sym_list[i],start='1900')
            if i % 10 == 0 :
                print('Got %i stocks for far' %(i))
                safe_database(db1,'close_2019_05_24.pkl')
        except Exception as e: 
            print(e)
            print('didnt get ' ,i )
            


