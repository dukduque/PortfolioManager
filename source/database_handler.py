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
path_to_output = os.path.abspath(os.path.join(parent_path,  'output'))

'''
Setup libraries
'''
import pandas as pd
import numpy as np
import multiprocessing as mp
import itertools
import pandas_datareader.data as web
import datetime
import pickle
import shutil 


def load_database(DB_file_name):
    '''
    Loads a pandas database stored as a pickle file
    Args:
        DB_file_name (str): name of the file
    '''
    path_to_database = os.path.join(path_to_data,DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists == False:
        return None
    
    try:
        return pd.read_pickle(path_to_database)
    except Exception as e: 
        print(e)
        
def save_database(BD,DB_file_name):
    '''
    Saves a database of in a pickle file. If a such file
    already exists, a copy of the old file is created.
    Args:
        DB (DataFrame): a pandas data frame
        DB_file_name (str): name of the file
    '''
    path_to_database = os.path.join(path_to_data,DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists:
        copy_name = 'copy_%s' %(DB_file_name)
        copy_path = os.path.join(path_to_data,copy_name)
        shutil.copyfile(path_to_database, copy_path)  
    BD.to_pickle(path_to_database)



def create_database(stock_symbol='GOOGLE', start = None, end = None):
    '''
    Creates a dataframe with one stock.
    Args:
        stock_symbol (str): stock symbol to query
        start (str or datetime): start date of the query
        end (str or datetime): end time of the query
    '''
    print(stock_symbol, start, end)
    db = web.DataReader(stock_symbol, 'yahoo', start=start, end=end)
    db = db.Close
    db = db.loc[~db.index.duplicated(keep='last')]
    db = db[db.index>=start]
    db.rename(stock_symbol, inplace=True)
    return db

def create_database_mp(input_date):
    return create_database(*input_date)
    
    

def add_stock(db, stock_symbol, start=None, end=None):
    '''
    Adds a stock to an existing dataframe.
    Args:
        db (DataFrame): current dataframe
    '''
    ndb = create_database(stock_symbol, start, end)
    return pd.concat((db,ndb),axis=1,join='outer')
    

def quotien_diff(x):
    '''
    Computes a division diff operation. Used to compute
    return out of stock prices.
    Args:
        x (DataSeries): pandas data series
    '''
    return np.divide(x[1:],x[:-1])
 

def get_returns(data_file = 'close_2019_05_24.pkl', start_date='2000', stocks = []):
    '''
    Computes the returns for stocks in the data file from
    a given year. All prices should be avaialbe to consider 
    a stock.
    Args:
        data_file (str): database file
        start_date (datetime): initial date
    '''
    assert start_date >= datetime.datetime(1970,1,1), 'Year should be from 1970'
    db = load_database(data_file)
    if len(stocks)>0:
        db = db[stocks]
    db = db[db.index>start_date]
    db = db.dropna(axis=1)
    db_r = db.apply(quotien_diff,axis=0) #compute returns
    db_r = db_r[db_r<1.5].dropna() #Filter outliers
    db = db.filter(db_r.columns, axis=1)
    db = db.filter(db_r.index, axis=0)
    
    r_data = np.array(db_r)
    return db, r_data

def update_database(db, n_proc=1):
    '''
    Updates a database from the last prices.
    If n_proc > 1, runs a mutiprocess version of
    the function to speedup the colection of data.
    '''
    ts = db.index[-1] #get last date in DB
    ndb = pd.DataFrame() 
    
    print('Updating %i stock with %i processors' %(len(db.columns), n_proc))
    if n_proc > 1:
        data_pool = mp.Pool(n_proc)
        stock_list = db.columns.to_list()
        n = 100
        chunks = [stock_list[i*n:(i+1)*n] for i in range((len(stock_list)+n-1)//n)]
        for chunk in chunks:
            stock_tasks = itertools.product(chunk,[ts])
            dbs = data_pool.map(create_database_mp,stock_tasks, 10)
            dbs = pd.concat(dbs,axis=1,join='outer')
            ndb = pd.concat((ndb, dbs),axis=1,join='outer')           
        data_pool.close()
    else:
        for c in db.columns:
            try:
                print('Updating ', c)
                ndb = add_stock(ndb, c, start=ts)
            except Exception as e: 
                print(e)
                
    out_db = pd.concat((db,ndb),axis=0,join='outer')
    out_db = out_db.loc[~out_db.index.duplicated(keep='last')]
    return out_db
    

        
 
def download_all_data(DB_file_name):
    symbols_df = web.get_nasdaq_symbols()
    symbols_df = symbols_df[symbols_df.ETF==False]
    sym_list = list(symbols_df.index)
    
    db1 = create_database(sym_list[0], start=1900)
    for i in range(1,len(sym_list)):
        try:
            db1 = add_stock(db1,sym_list[i],start='1900')
            if i % 100 == 0 :
                cols = len(db1.columns)
                print('Got %i stocks for far' %(cols))
                save_database(db1,DB_file_name)
        except Exception as e: 
            print(e)

    
    


if __name__ == '__main__':
    db = load_database('close_2019_05_25.pkl')
    db = update_database(db, 4)
    save_database(db, 'close_2019_05_25.pkl')
