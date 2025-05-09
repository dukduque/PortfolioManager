"""
Created on Thu May 23 22:32:31 2019

@author: dduque

This module implements function to manage the prices database and
access the information. As of 2019/06/03, the package to access price
data is yfinance.
"""

import enum
import shutil
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
path_to_data = os.path.abspath(os.path.join(parent_path, "data"))
from source import util

EMPTY_METADATA = {
    "name": "",
    "sector": "",
    "subsector": "",
    "market_cap": "",
}


def set_data_path(new_path_to_data):
    path_is_new = False
    new_path = Path(new_path_to_data)
    if not new_path.is_dir():
        try:
            path_is_new = True
            os.mkdir(str(new_path))
        except Exception as e:
            raise f"{new_path} is no a valid path. {str(e)}"

    global path_to_data
    path_to_data = new_path.absolute()
    return path_is_new


class DataManager:
    """
    An instance of this class access and maintains the data.
    """

    def __init__(self, db_file="close.pkl", metadata_file="metadata.pkl"):
        self.db = load_database(db_file)
        self.metadata = get_tickers_metadata(metadata_file)
        self.db_file = db_file
        self.metadata_file = metadata_file

    def get_prices(self, assets=None):
        """
        assets (str or list): an asset or list of assets.
        """
        if assets is None or len(assets) == 0:
            return self.db

        if type(assets) == str:
            assets = [assets]

        for asset in assets:
            if asset not in self.db.columns:
                self.db = update_database_single_stock(
                    self.db, asset, self.db_file, self.metadata_file
                )
        df_out = pd.DataFrame(index=self.db.index, columns=assets)
        df_out.update(self.db)
        return df_out

    def get_metadata(self, asset):
        assert type(asset) == str
        if asset in self.metadata:
            return self.metadata[asset]
        try:
            asset_ticket = yf.Ticker(asset)
            info = asset_ticket.info
            self.metadata[asset] = EMPTY_METADATA.copy()
            self.metadata[asset]["name"] = info["shortName"]
            self.metadata[asset]["quoteType"] = info["quoteType"]
            if info["quoteType"] == "ETF":
                self.metadata[asset]["sector"] = "ETF"
                self.metadata[asset]["industry"] = "ETF"
            elif info["quoteType"] == "EQUITY":
                self.metadata[asset]["sector"] = info["sector"]
                self.metadata[asset]["industry"] = info["industry"]
        except Exception:
            self.metadata[asset] = EMPTY_METADATA
            self.metadata[asset]["name"] = asset
        save_metadata(self.metadata, self.metadata_file)
        return self.metadata[asset]

    def get_returns(self, start_date, end_date, stocks=[], outlier_return=10):
        """
        Computes returns from specific dates and list of securities.
        """
        if len(stocks) == 0 and hasattr(self, "_returns"):
            return self._returns

        for s in stocks:
            if s not in self.db.columns:
                try:
                    self.get_prices(s)
                    self.get_metadata(s)
                except Exception:
                    print(f"Fail while obtaining data from security {s}")
        db = self.db[self.db.index >= start_date]
        db = db[db.index <= end_date]
        db = db.dropna(axis=0, how="all")
        db = db.dropna(axis=1)
        db_r = db.apply(quotien_diff, axis=0)  # compute returns
        db_r = db_r[db_r < outlier_return].dropna(axis=0)  # Filter outliers

        self._returns = db_r
        return db_r

    def repair_data(self, stock_symbol, start_date):
        if stock_symbol not in self.db.columns:
            return
        stock = yf.Ticker(stock_symbol)
        trials = 3
        for repairing_date in reversed(self.db.index):
            if repairing_date < start_date:
                break
            if np.isnan(self.db.loc[repairing_date, stock_symbol]):
                time.sleep(np.random.uniform(0, 0.1))
                update = stock.history(
                    start=repairing_date,
                    end=repairing_date + dt.timedelta(days=1),
                ).Close
                if (
                    repairing_date in update.index
                    and update.loc[repairing_date] != np.nan
                ):
                    self.db.loc[repairing_date, stock_symbol] = update.loc[
                        repairing_date
                    ]
                else:
                    print("Not repaired", stock_symbol, repairing_date)
                    trials -= 1
                    if trials == 0:
                        print("Still nan")
                        break

        save_database(self.db, self.db_file)

    @property
    def securities(self):
        return list(self.db.columns)


def save_sp500_tickers():
    """
    https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    """
    resp = requests.get(
        "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = {}
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text.replace("\n", "")
        ticker_info = {}
        ticker_info["name"] = row.findAll("td")[1].text.replace("\n", "")
        ticker_info["sector"] = row.findAll("td")[3].text.replace("\n", "")
        ticker_info["subsector"] = row.findAll("td")[4].text.replace("\n", "")
        tickers[ticker] = ticker_info
    for ticker in tickers:
        _, mkt_cap = get_market_cap(ticker)
        tickers[ticker]["market_cap"] = mkt_cap
    path_to_file = os.path.join(path_to_data, "sp500tickers.pickle")
    with open(path_to_file, "wb") as f:
        pickle.dump(tickers, f)


def get_market_cap(ticker_str):
    """
    Retrieves the market cap of the ticker given as input. If the ticker information is not available, returns zero.
    
    Note: We add a random sleep to avoid overloading the Yahoo Finance API.
    """
    ticker = yf.Ticker(ticker_str)
    print(f"Getting {ticker_str} market cap...")
    try:
        # Add a random sleep to avoid overloading the Yahoo Finance API
        time.sleep(np.random.uniform(0, 500) / 1000)
        # Return the ticker and its market capitalization
        return (ticker_str, ticker.info["marketCap"])
    except Exception as e:
        # If there is an error (e.g. the ticker is not recognized), print the error message
        print(e)
        # Return the ticker and a market capitalization of 0.0
        return (ticker_str, 0.0)


def get_sp500_tickers():
    path_to_file = Path(os.path.join(path_to_data, "sp500.pkl"))
    if not path_to_file.exists:
        save_sp500_tickers()
    return pickle.load(path_to_file.open("rb"))


def get_tickers_metadata(meta_data_file):
    path_to_file = Path(os.path.join(path_to_data, meta_data_file))
    if not path_to_file.exists:
        return {}
    return pickle.load(path_to_file.open("rb"))


def save_rusell1000_tickers():
    """
    https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    """
    resp = requests.get("https://en.wikipedia.org/wiki/Russell_1000_Index")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    tables = soup.find_all("table")
    table = tables[2]
    tickers = []
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[1].text
        tickers.append(ticker.replace("\n", ""))
    path_to_file = os.path.join(path_to_data, "rusell1000tickers.pickle")
    with open(path_to_file, "wb") as f:
        pickle.dump(tickers, f)

    return tickers


def load_database(db_file_name):
    """
    Loads a pandas database stored as a pickle file
    Args:
        db_file_name (str): name of the file
    """
    path_to_database = os.path.join(path_to_data, db_file_name)
    exists = os.path.isfile(path_to_database)
    if not exists:
        raise "File %s does not exist" % (db_file_name)
    try:
        return pd.read_pickle(path_to_database)
    except Exception as e:
        print(e)
    return None


def save_database(BD, db_file_name):
    """
    Saves a database of in a pickle file. If a such file
    already exists, a copy of the old file is created.
    Args:
        DB (DataFrame): a pandas data frame
        db_file_name (str): name of the file
    """
    path_to_database = os.path.join(path_to_data, db_file_name)
    exists = os.path.isfile(path_to_database)
    if exists:
        copy_name = "copy_%s" % (db_file_name)
        copy_path = os.path.join(path_to_data, copy_name)
        shutil.copyfile(path_to_database, copy_path)

    try:
        BD.to_pickle(path_to_database)
        return True
    except Exception as e:
        print(e)
    return False


def save_metadata(metadata, metadata_file):
    path_to_database = os.path.join(path_to_data, metadata_file)
    print(path_to_database)
    exists = os.path.isfile(path_to_database)
    if exists:
        copy_name = "copy_%s" % (metadata_file)
        copy_path = os.path.join(path_to_data, copy_name)
        shutil.copyfile(path_to_database, copy_path)
    with open(path_to_database, "wb") as handle:
        pickle.dump(metadata, handle, pickle.HIGHEST_PROTOCOL)


def create_database(stock_symbol, start=None, end=None):
    """
    Creates a dataframe with one stock.
    Args:
        stock_symbol (str): stock symbol to query
        start (str or datetime): start date of the query
        end (str or datetime): end time of the query (if str, this is a
                               exclusive interval)
    Return:
        db (DataFrame): a dataframe with the requested symbol
        status (bool): true if the query was successful
    """
    try:
        time.sleep(np.random.uniform(0, 0.1))
        db = yf.download(stock_symbol, start=start, end=end, threads=False)
        if len(db.index) == 0:
            # No data found
            return stock_symbol, None, False

        db = db.Close
        db = db.loc[~db.index.duplicated(keep="last")]
        if start is not None:
            db = db[db.index >= start]
        # db.rename(stock_symbol, inplace=True)
        return stock_symbol, db, True
    except Exception as e:
        print(e)
        print(f"Failed to get: {stock_symbol} , {start}, {end}")

    return stock_symbol, None, False


def create_database_mp(input_date):
    return create_database(*input_date)


def add_stock(db, stock_symbol, start=None, end=None):
    """
    Adds a stock to an existing dataframe.
    Args:
        db (DataFrame): current dataframe
    """
    _, ndb, status = create_database(stock_symbol, start, end)
    if status:
        join_type = "inner" if len(db.index) > 0 else "outer"
        return pd.concat((db, ndb), axis=1, join=join_type), True
    else:
        return db, False


def quotien_diff(x):
    """
    Computes a division diff operation. Used to compute
    return out of stock prices.
    Args:
        x (DataSeries): pandas data series
    """
    y = np.array(x)
    return pd.Series(y[1:] / y[:-1], index=x[1:].index)


def get_returns(
    data_file,
    start_date="2000",
    end_date=dt.datetime.today(),
    stocks=[],
    outlier_return=10,
):
    """
    Computes the returns for stocks in the data file from
    a given year. All prices should be available to consider
    a stock.
    Args:
        data_file (str): database file
        start_date (datetime): initial date
    Return:
        db (DataFrame): dataframe with the stock prices
        db_r (DataFrame): dataframe with the returns
    """
    assert data_file is None, "Deprecated function"
    assert start_date >= datetime.datetime(
        1970, 1, 1
    ), "Year should be from 1970"
    db = load_database(data_file)
    if len(stocks) > 0:
        db = db[db.columns.intersection(stocks)]
    db = db[db.index >= start_date]
    db = db[db.index <= end_date]
    db = db.dropna(axis=0, how="all")
    db = db.dropna(axis=1)
    db_r = db.apply(quotien_diff, axis=0)  # compute returns
    db_r = db_r[db_r < outlier_return].dropna(axis=1)  # Filter outliers
    db = db.filter(db_r.columns, axis=1)
    db = db.filter(db_r.index, axis=0)

    return db, db_r


class StockUpdateStatus(enum.Enum):
    OK = 0
    NOT_FOUND = 1
    FAILED = 2


def update_stock_prices(stock_series, retries=3, backoff_seconds=1.0):
    ticker_name = stock_series.name
    stock_nan = stock_series.isna()
    start_date = stock_series[~stock_nan].index.max() + dt.timedelta(days=1)
    end_date = stock_series.index.max() + dt.timedelta(days=1)
    if start_date >= end_date:
        return StockUpdateStatus.OK
    for i in range(retries):
        try:
            new_data = yf.download(stock_series.name, start=start_date,
                                   end=end_date, threads=False,
                                   multi_level_index=False).Close
            if len(new_data.index) == 0:
                return StockUpdateStatus.NOT_FOUND
            new_data = new_data.rename(ticker_name, inplace=True)
            stock_series.update(new_data)
            return StockUpdateStatus.OK
        except Exception as e:
            print(f"Failed to get: {stock_series.name} in retry {i} with {e}")
        sleep_time = (i + backoff_seconds) * 2 + np.random.uniform(0, 0.1)
        time.sleep(sleep_time)
    return StockUpdateStatus.FAILED


def update_database(db):
    """
    Updates a database from the last prices.
    If n_proc > 1, runs a mutiprocess version of
    the function to speedup the colection of data.
    """
    end_date = max(db.index.max(), dt.datetime.today())
    new_date_range = pd.date_range(start=db.index.max(),
                                   end=end_date, freq="D")
    db = db.reindex(db.index.union(
        new_date_range[np.array([x.weekday() < 5 for x in new_date_range])]))
    failed_updates = []
    # Sort the columns by the number of missing valuesa and fix the ones
    # with the most number of missing values.
    sorted_columns = db.isna().sum(axis=0).sort_values(ascending=False).index
    for c in sorted_columns:
        download_status = update_stock_prices(db[c])
        if download_status == StockUpdateStatus.NOT_FOUND:
            # If the stock is not found, we can't remove it because it might
            # exist in some portfolio at some point in the past.
            continue
        elif download_status == StockUpdateStatus.FAILED:
            failed_updates.append(c)
    print("Failed to update %i stocks" % len(failed_updates))
    # Drop rows where all values are missing (e.g., weekends)
    db = db[db.isna().sum(axis=1) < len(db.columns)]
    # Drop columns where the last `days_back` values are missing.
    _unlisted_days_threshold = 2500
    db = db.loc[:, ~db.isna().iloc[-_unlisted_days_threshold:].all()]
    return db


def update_database_single_stock(
    db,
    ticker_symbol,
    db_output_file="close.pkl",
    info_output_file="assets_listing.pkl",
):
    # TODO: Modify this function to use the new update function
    # that already has retries.
    db, status = add_stock(db, ticker_symbol, db.index[0], dt.datetime.today())
    if status:
        save_database(db, db_output_file)
    else:
        print(f"Database was not updated with ticker {ticker_symbol}")
    return db


def download_all_data(
    db_file_name,
    tickers=[],
    sp500=False,
    rusell1000=False,
    include_bonds=False,
    n_proc=4,
):
    stocks = set()
    stocks.update(tickers)
    if sp500:
        sp500_stocks = save_sp500_tickers()
        stocks.update(sp500_stocks.keys())
    if rusell1000:
        rusell1000_stocks = save_rusell1000_tickers()
        stocks.update(rusell1000_stocks)
    if include_bonds:
        # TODO: find a larger list of bonds and/or bonds ETFs
        stocks.update(["GOVT", "BLV"])

    stocks = list(stocks)
    ini_data = dt.datetime(year=2000, month=1, day=1)
    today = dt.datetime.today()
    data = yf.download(stocks, start=ini_data, end=today, threads=n_proc)
    close_data = data.Close
    save_database(close_data, db_file_name)
    return close_data


def run_update_process(
    db_file_in="close.pkl", db_file_out="close.pkl", n_proc=4, days_back=1
):
    db = load_database(db_file_in)
    print(f"Loading db with {len(db.columns)} stocks")
    db = update_database(db)
    print(f"Updating db with {len(db.columns)} stocks")
    save_database(db, db_file_out)


if __name__ == "__main__":
    args = util.dh_parse_arguments()
    print(args)
    if args.a == "u":
        today_ts = datetime.datetime.today()
        str_today = str(today_ts)
        out_file = "close.pkl"  # % (str_today.split(' ')[0])
        run_update_process(args.db_file, out_file, args.n_proc, args.days_back)
    elif args.a == "d":
        download_all_data(args.db_file, sp500=True, n_proc=args.n_proc)
    elif args.a == "sp500":
        save_sp500_tickers()
