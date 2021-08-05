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
from numpy.testing._private.utils import raises
import yfinance as yf
import requests
import pickle
import datetime
import datetime as dt
import itertools
import bs4 as bs
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import os
import time
from pathlib import Path
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.insert(0, parent_path)
path_to_data = os.path.abspath(os.path.join(parent_path, 'data'))

from source import util

EMPTY_METADATA = {
    'name': '',
    'sector': '',
    'subsector': '',
    'market_cap': '',
}


def set_data_path(new_path_to_data):
    new_path = Path(new_path_to_data)
    if (not new_path.is_dir()):
        raises(f'{new_path} is no a valid path.')
    global path_to_data
    path_to_data = new_path.absolute()


class DataManager:
    """
    An instance of this class access and maintains the data.
    """
    def __init__(self, db_file='close.pkl', metadata_file='metadata.pkl'):
        self.db = load_database(db_file)
        self.metadata = get_tickers_metadata(metadata_file)
        self.db_file = db_file
        self.metadata_file = metadata_file
    
    def get_prices(self, assets=None):
        '''
        assets (str or list): an asset or list of assets.
        '''
        if assets is None or len(assets) == 0:
            return self.db
        
        if type(assets) == str:
            assets = [assets]
        
        for asset in assets:
            if asset not in self.db.columns:
                self.db = update_database_single_stock(self.db, asset,
                                                       self.db_file,
                                                       self.metadata_file)
        return self.db[assets]
    
    def get_metadata(self, asset):
        assert type(asset) == str
        if asset in self.metadata:
            return self.metadata[asset]
        try:
            asset_ticket = yf.Ticker(asset)
            info = asset_ticket.info
            self.metadata[asset] = EMPTY_METADATA.copy()
            self.metadata[asset]['name'] = info['shortName']
            self.metadata[asset]['quoteType'] = info['quoteType']
            if info['quoteType'] == "ETF":
                self.metadata[asset]['sector'] = 'ETF'
                self.metadata[asset]['industry'] = 'ETF'
            elif info['quoteType'] == "EQUITY":
                self.metadata[asset]['sector'] = info['sector']
                self.metadata[asset]['industry'] = info['industry']
        except Exception:
            self.metadata[asset] = EMPTY_METADATA
            self.metadata[asset]['name'] = asset
        safe_metadata(self.metadata, self.metadata_file)
        return self.metadata[asset]
    
    def get_returns(self, start_date, end_date, stocks=[], outlier_return=10):
        '''
        Computes returns from specific dates and list of securities.
        '''
        if len(stocks) == 0 and hasattr(self, '_returns'):
            return self._returns
        
        for s in stocks:
            if s not in self.db.columns:
                try:
                    self.get_prices(s)
                    self.get_metadata(s)
                except Exception:
                    print(f'Fail while obtaining data from security {s}')
        db = self.db[self.db.index >= start_date]
        db = db[db.index <= end_date]
        db = db.dropna(axis=0, how='all')
        db = db.dropna(axis=1)
        db_r = db.apply(quotien_diff, axis=0)  # compute returns
        db_r = db_r[db_r < outlier_return].dropna(axis=0)  # Filter outliers
        
        self._returns = db_r
        return db_r
    
    @property
    def securities(self):
        return list(self.db.columns)


def save_sp500_tickers():
    '''
    https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    '''
    resp = requests.get(
        'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = {}
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.replace('\n', '')
        ticker_info = {}
        ticker_info['name'] = row.findAll('td')[1].text.replace('\n', '')
        ticker_info['sector'] = row.findAll('td')[3].text.replace('\n', '')
        ticker_info['subsector'] = row.findAll('td')[4].text.replace('\n', '')
        tickers[ticker] = ticker_info
    for ticker in tickers:
        _, mkt_cap = get_market_cap(ticker)
        tickers[ticker]['market_cap'] = mkt_cap
    path_to_file = os.path.join(path_to_data, "sp500tickers.pickle")
    with open(path_to_file, "wb") as f:
        pickle.dump(tickers, f)


def get_market_cap(ticker_str):
    '''
    Retrieves the market cap of the ticker given as input. If the ticker information is not available, returns zero.
    '''
    ticker = yf.Ticker(ticker_str)
    print(f'Getting {ticker_str} market cap...')
    try:
        time.sleep(np.random.uniform(0, 500) / 1000)
        return (ticker_str, ticker.info['marketCap'])
    except Exception as e:
        print(e)
        return (ticker_str, 0.0)


def get_sp500_tickers():
    path_to_file = Path(os.path.join(path_to_data, "sp500.pkl"))
    if not path_to_file.exists:
        save_sp500_tickers()
    return pickle.load(path_to_file.open('rb'))


def get_tickers_metadata(meta_data_file):
    path_to_file = Path(os.path.join(path_to_data, meta_data_file))
    if not path_to_file.exists:
        return {}
    return pickle.load(path_to_file.open('rb'))


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
    if not exists:
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


def safe_metadata(metadata, metadata_file):
    path_to_database = os.path.join(path_to_data, metadata_file)
    print(path_to_database)
    exists = os.path.isfile(path_to_database)
    if exists:
        copy_name = 'copy_%s' % (metadata_file)
        copy_path = os.path.join(path_to_data, copy_name)
        shutil.copyfile(path_to_database, copy_path)
    with open(path_to_database, 'wb') as handle:
        pickle.dump(metadata, handle, pickle.HIGHEST_PROTOCOL)


def create_database(stock_symbol, start=None, end=None):
    '''
    Creates a dataframe with one stock.
    Args:
        stock_symbol (str): stock symbol to query
        start (str or datetime): start date of the query
        end (str or datetime): end time of the query (if None, databes includes today's data)
    Return:
        db (DataFrame): a dataframe with the requested symbol
        status (bool): true if the query was succesfull
    '''
    print(stock_symbol, start, end)
    
    try:
        stock = yf.Ticker(stock_symbol)
        tomorow = datetime.datetime.today() + datetime.timedelta(days=1)
        _end_date = end if end is not None else datetime.datetime.today()
        db = yf.download(stock_symbol, start=start, end=end, threads=False)
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
    y = np.array(x)
    return pd.Series(y[1:] / y[:-1], index=x[1:].index)


def get_returns(data_file,
                start_date='2000',
                end_date=dt.datetime.today(),
                stocks=[],
                outlier_return=10):
    '''
    Computes the returns for stocks in the data file from
    a given year. All prices should be avaialbe to consider
    a stock.
    Args:
        data_file (str): database file
        start_date (datetime): initial date
    Return:
        db (DataFrame): dataframe with the stock prices
        db_r (DataFrame): dataframe with the returns
    '''
    assert data_file is None, 'Deprecated function'
    assert start_date >= datetime.datetime(1970, 1,
                                           1), 'Year should be from 1970'
    db = load_database(data_file)
    if len(stocks) > 0:
        db = db[db.columns.intersection(stocks)]
    db = db[db.index >= start_date]
    db = db[db.index <= end_date]
    db = db.dropna(axis=0, how='all')
    db = db.dropna(axis=1)
    db_r = db.apply(quotien_diff, axis=0)  # compute returns
    db_r = db_r[db_r < outlier_return].dropna(axis=1)  # Filter outliers
    db = db.filter(db_r.columns, axis=1)
    db = db.filter(db_r.index, axis=0)
    
    return db, db_r


def update_database(db, n_proc, days_back):
    '''
    Updates a database from the last prices.
    If n_proc > 1, runs a mutiprocess version of
    the function to speedup the colection of data.
    '''
    ts = db.index[
        -days_back]  # get last date in DB  #TODO: what if last date was NaN
    ndb = pd.DataFrame()
    failed_stocks = []
    print('Updating %i stock with %i processors' % (len(db.columns), n_proc))
    if n_proc > 1:
        data_pool = mp.Pool(n_proc)
        stock_list = db.columns.to_list()
        n = 100
        chunks = [
            stock_list[i * n:(i + 1) * n]
            for i in range((len(stock_list) + n - 1) // n)
        ]
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
                if not status:  # No updated was performed
                    failed_stocks.append(c)
            except Exception as e:
                print(e)
    print(failed_stocks)
    # Create new rows with NaN values
    for new_date in ndb.index:
        print(f'Crete row for {new_date}')
        if new_date not in db.index:
            db.loc[new_date] = [np.nan] * len(db.columns)
    # Update NaN values
    db.update(ndb, overwrite=False)
    return db


def update_database_single_stock(db,
                                 ticker_symbol,
                                 db_output_file='close.pkl',
                                 info_output_file='assets_listing.pkl'):
    
    db, status = add_stock(db, ticker_symbol, db.index[0], dt.datetime.today())
    if status:
        save_database(db, db_output_file)
    else:
        print(f"Database was not updated with ticker {ticker_symbol}")
    return db


def download_all_data(DB_file_name,
                      sp500=True,
                      rusell1000=False,
                      include_bonds=True,
                      n_proc=4):
    # symbols_df = web.get_nasdaq_symbols()
    # symbols_df = symbols_df[symbols_df.ETF == False]
    # symbols_df = symbols_df[symbols_df.ETF == False]
    # sym_list = list(symbols_df.index)
    
    sp500_stocks = save_sp500_tickers()
    rusell1000_stocks = save_rusell1000_tickers()
    stocks = set()
    if sp500:
        stocks.update(sp500_stocks.keys())
    if rusell1000:
        stocks.update(rusell1000_stocks)
    if include_bonds:
        # TODO: find a larger list of bonds and/or bonds ETFs
        stocks.update(["GOVT", "BLV"])
    
    stocks = list(stocks)
    ini_data = dt.datetime(year=2000, month=1, day=1)
    today = dt.datetime.today()
    data = yf.download(stocks, start=ini_data, end=today, threads=n_proc)
    close_data = data.Close
    save_database(close_data, DB_file_name)
    return close_data


def run_update_process(db_file_in='close.pkl',
                       db_file_out='close.pkl',
                       n_proc=4,
                       days_back=1):
    db = load_database(db_file_in)
    db = update_database(db, n_proc, days_back)
    save_database(db, db_file_out)


if __name__ == '__main__':
    args = util.dh_parse_arguments()
    print(args)
    if args.a == 'u':
        today_ts = datetime.datetime.today()
        str_today = str(today_ts)
        out_file = 'close.pkl'  # % (str_today.split(' ')[0])
        run_update_process(args.db_file, out_file, args.n_proc, args.days_back)
    elif args.a == 'd':
        download_all_data(args.db_file, n_proc=args.n_proc)
    elif args.a == 'sp500':
        save_sp500_tickers()
