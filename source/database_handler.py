#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:32:31 2019

@author: dduque

This module implements function to manage the prices database and
acces the information. As of 2019/06/03, the package to acces price
data is yfinance.
"""

'''
Setup paths
'''


import shutil
from alpha_vantage.timeseries import TimeSeries  # API limits, 5 queries per minute
import pandas_datareader.data as web  # yahoo or av-daily, but glichy
import yfinance as yf  # Works awsome!
import requests
import pickle
import datetime
import itertools
import bs4 as bs
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import os
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.append(parent_path)
path_to_data = os.path.abspath(os.path.join(parent_path, 'data'))
path_to_output = os.path.abspath(os.path.join(parent_path, 'output'))

import source.popt_utils as popt_utils
'''
Setup libraries
'''
AV_TS = TimeSeries(key='OSZLY662JJE9SVS1', output_format='pandas')


def save_sp500_tickers():
    '''
    https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    '''
    resp = requests.get(
        'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker.replace('\n', ''))
    path_to_file = os.path.join(path_to_data, "sp500tickers.pickle")
    with open(path_to_file, "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def save_rusell1000_tickers():
    '''
    https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    '''
    resp = requests.get('https://en.wikipedia.org/wiki/Russell_1000_Index')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    tables = soup.find_all('table')
    table = tables[2]
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker.replace('\n', ''))
    path_to_file = os.path.join(path_to_data, "rusell1000tickers.pickle")
    with open(path_to_file, "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def load_database(DB_file_name):
    '''
    Loads a pandas database stored as a pickle file
    Args:
        DB_file_name (str): name of the file
    '''
    path_to_database = os.path.join(path_to_data, DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists == False:
        raise 'File %s does not exist' % (DB_file_name)
    try:
        return pd.read_pickle(path_to_database)
    except Exception as e:
        print(e)


def save_database(BD, DB_file_name):
    '''
    Saves a database of in a pickle file. If a such file
    already exists, a copy of the old file is created.
    Args:
        DB (DataFrame): a pandas data frame
        DB_file_name (str): name of the file
    '''
    path_to_database = os.path.join(path_to_data, DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists:
        copy_name = 'copy_%s' % (DB_file_name)
        copy_path = os.path.join(path_to_data, copy_name)
        shutil.copyfile(path_to_database, copy_path)
    BD.to_pickle(path_to_database)


def create_database(stock_symbol='GOOGLE', start=None, end=None):
    '''
    Creates a dataframe with one stock.
    Args:
        stock_symbol (str): stock symbol to query
        start (str or datetime): start date of the query
        end (str or datetime): end time of the query
    Return:
        db (DataFrame): a dataframe with the requested symbol
        status (bool): true if the query was succesfull
    '''
    print(stock_symbol, start, end)

    try:
        #        db = web.DataReader(stock_symbol, "av-daily", start=start ,end=end, access_key='OSZLY662JJE9SVS1')
        #        db['date'] =  [dt.datetime.strptime(d, '%Y-%m-%d') for d in db.index]
        #        db = db.set_index('date')
        #        db = db.close

        #        db = web.DataReader(stock_symbol, 'yahoo', start=start, end=end)
        #        db = db.Close

        stock = yf.Ticker(stock_symbol)
        db = stock.history(start=start, end=end)
        db = db.Close

        db = db.loc[~db.index.duplicated(keep='last')]
        db = db[db.index >= start]
        db.rename(stock_symbol, inplace=True)
        return stock_symbol, db, True
    except Exception as e:
        print(e)
        print('Fail to get: ', stock_symbol, start, end)

    return stock_symbol, None, False


def create_database_mp(input_date):
    return create_database(*input_date)


def add_stock(db, stock_symbol, start=None, end=None):
    '''
    Adds a stock to an existing dataframe.
    Args:
        db (DataFrame): current dataframe
    '''
    _, ndb, status = create_database(stock_symbol, start, end)
    if status:
        return pd.concat((db, ndb), axis=1, join='outer'), True
    else:
        return db, False


def quotien_diff(x):
    '''
    Computes a division diff operation. Used to compute
    return out of stock prices.
    Args:
        x (DataSeries): pandas data series
    '''
    return np.divide(x[1:], x[:-1])


def get_returns(data_file, start_date='2000', stocks=[]):
    '''
    Computes the returns for stocks in the data file from
    a given year. All prices should be avaialbe to consider
    a stock.
    Args:
        data_file (str): database file
        start_date (datetime): initial date
    '''
    assert start_date >= datetime.datetime(
        1970, 1, 1), 'Year should be from 1970'
    db = load_database(data_file)
    if len(stocks) > 0:
        db = db[db.columns.intersection(stocks)]
    db = db[db.index > start_date]
    db = db.dropna(axis=0, how='all')
    db = db.dropna(axis=1)
    db_r = db.apply(quotien_diff, axis=0)  # compute returns
    db_r = db_r[db_r < 10.0].dropna(axis=1)  # Filter outliers
    db = db.filter(db_r.columns, axis=1)
    db = db.filter(db_r.index, axis=0)

    r_data = np.array(db_r)
    return db, r_data


def cov_estimation():
    pass


def update_database(db, n_proc=1):
    '''
    Updates a database from the last prices.
    If n_proc > 1, runs a mutiprocess version of
    the function to speedup the colection of data.
    '''
    ts = db.index[-1]  # get last date in DB
    ndb = pd.DataFrame()
    failed_stocks = []
    print('Updating %i stock with %i processors' % (len(db.columns), n_proc))
    if n_proc > 1:
        data_pool = mp.Pool(n_proc)
        stock_list = db.columns.to_list()
        n = 100
        chunks = [stock_list[i*n:(i+1)*n]
                  for i in range((len(stock_list)+n-1)//n)]
        for chunk in chunks:
            stock_tasks = itertools.product(chunk, [ts])
            mp_out = data_pool.map(create_database_mp, stock_tasks)
            for s, db_s, status_s in mp_out:
                if status_s:
                    ndb = pd.concat((ndb, db_s), axis=1, join='outer')
                else:
                    failed_stocks.append(s)

        data_pool.close()
    else:
        for c in db.columns:
            try:
                ndb, status = add_stock(ndb, c, start=ts)
                if status == False:  # No updated was performed
                    failed_stocks.append(c)
            except Exception as e:
                print(e)
    print(failed_stocks)
    out_db = pd.concat((db, ndb), axis=0, join='outer')
    out_db = out_db.loc[~out_db.index.duplicated(keep='last')]
    return out_db


def download_all_data(DB_file_name):
    symbols_df = web.get_nasdaq_symbols()
    symbols_df = symbols_df[symbols_df.ETF == False]
    sym_list = list(symbols_df.index)

    db1, _ = create_database(sym_list[0], start=1900)
    for i in range(1, len(sym_list)):
        try:
            db1 = add_stock(db1, sym_list[i], start='1900')
            if i % 100 == 0:
                cols = len(db1.columns)
                print('Got %i stocks for far' % (cols))
                save_database(db1, DB_file_name)
        except Exception as e:
            print(e)
    return db1

def run_update_process(db_file_in='close.pkl', db_file_out='close.pkl', n_proc=4):
    db = load_database(db_file_in)
    db = update_database(db, n_proc)
    save_database(db, db_file_out)

if __name__ == '__main__':
    args = popt_utils.dh_parse_arguments()
    print(args)
    if args.a == 'u':
        today_ts = datetime.datetime.today()
        str_today = str(today_ts)
        out_file = 'close_%s.pkl' % (str_today.split(' ')[0])
        run_update_process(arg.db_file, out_file, arg.proc)

    #db = load_database('close_2019-05-26.pkl')
    #ini_time = datetime.datetime(2019,5,20)
    #new_db = yf.download(db.columns.to_list(), start = ini_time)
    #db = update_database(db, 20)
    #save_database(db, 'close_2019_05_26.pkl')

# for stock_symbol in db.columns[:10]:
#    stock = yf.Ticker(stock_symbol)
#    s_data = stock.history('5d')
#    print(s_data.head())
#
#
# if False:
#
#    # Get json object with the intraday data and another with  the call's metadata
#    data, meta_data = ts.get_daily('GOOGL')
