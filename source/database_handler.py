"""
Created on Thu May 23 22:32:31 2019

@author: dduque

This module implements function to manage the prices database and
access the information. As of 2019/06/03, the package to access price
data is yfinance.
"""

import enum
import logging
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
from data.base import EMPTY_METADATA, quotient_diff as quotien_diff  # noqa: F401

log = logging.getLogger(__name__)


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


# DataManager is the backward-compatible alias for the local pickle workflow.
# New code should import from data.local or data.alpaca directly.
from data.local import LocalDataManager as DataManager  # noqa: F401


def save_sp500_tickers():
    """
    Fetch S&P 500 constituents from Wikipedia and cache them locally.

    Returns a dict: {ticker: {name, sector, subsector}}
    Tickers are normalised to Alpaca format (dots replaced with slashes,
    e.g. BRK.B -> BRK/B).

    https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
    """
    resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (compatible; portfolio-manager/1.0)"},
        timeout=15,
    )
    resp.raise_for_status()
    soup = bs.BeautifulSoup(resp.text, "lxml")
    # Primary selector: Wikipedia table has id="constituents"
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        table = soup.find("table", {"class": "wikitable sortable"})
    if table is None:
        raise RuntimeError("Could not find S&P 500 constituents table on Wikipedia.")

    tickers = {}
    for row in table.findAll("tr")[1:]:
        cells = row.findAll("td")
        if len(cells) < 5:
            continue
        ticker = cells[0].text.strip()
        tickers[ticker] = {
            "name":      cells[1].text.strip(),
            "sector":    cells[3].text.strip(),
            "subsector": cells[4].text.strip(),
        }

    os.makedirs(path_to_data, exist_ok=True)
    path_to_file = os.path.join(path_to_data, "sp500tickers.pickle")
    with open(path_to_file, "wb") as f:
        pickle.dump(tickers, f)
    log.info("Saved %d S&P 500 tickers to %s", len(tickers), path_to_file)
    return tickers


def get_sp500_tickers():
    """Return cached S&P 500 dict, fetching from Wikipedia if not cached."""
    path_to_file = Path(os.path.join(path_to_data, "sp500tickers.pickle"))
    if not path_to_file.exists():
        return save_sp500_tickers()
    return pickle.load(path_to_file.open("rb"))


def get_market_cap(ticker_str):
    """
    Retrieves the market cap of the ticker given as input. If the ticker information is not available, returns zero.
    
    Note: We add a random sleep to avoid overloading the Yahoo Finance API.
    """
    ticker = yf.Ticker(ticker_str)
    log.debug("Getting %s market cap.", ticker_str)
    try:
        # Add a random sleep to avoid overloading the Yahoo Finance API
        time.sleep(np.random.uniform(0, 500) / 1000)
        # Return the ticker and its market capitalization
        return (ticker_str, ticker.info["marketCap"])
    except Exception as e:
        log.warning("get_market_cap(%s) failed: %s", ticker_str, e)
        return (ticker_str, 0.0)




def get_tickers_metadata(meta_data_file):
    path_to_file = Path(os.path.join(path_to_data, meta_data_file))
    if not path_to_file.exists():
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
        log.warning("Failed to read pickle %s: %s", path_to_database, e)
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
        log.warning("Failed to save database to %s: %s", path_to_database, e)
    return False


def save_metadata(metadata, metadata_file):
    path_to_database = os.path.join(path_to_data, metadata_file)
    log.debug("Saving metadata to %s", path_to_database)
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
        log.warning("Failed to get %s (%s – %s): %s", stock_symbol, start, end, e)

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


# quotien_diff is now imported from data.base as an alias above.


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
            log.warning("Failed to get %s on retry %d: %s", stock_series.name, i, e)
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
    if failed_updates:
        log.warning("Failed to update %d stock(s): %s", len(failed_updates), failed_updates)
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
        log.warning("Database was not updated with ticker %s.", ticker_symbol)
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
    log.info("Loaded db with %d stocks.", len(db.columns))
    db = update_database(db)
    log.info("Updated db with %d stocks.", len(db.columns))
    save_database(db, db_file_out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    args = util.dh_parse_arguments()
    log.debug("CLI args: %s", args)
    if args.a == "u":
        today_ts = datetime.datetime.today()
        str_today = str(today_ts)
        out_file = "close.pkl"  # % (str_today.split(' ')[0])
        run_update_process(args.db_file, out_file, args.n_proc, args.days_back)
    elif args.a == "d":
        download_all_data(args.db_file, sp500=True, n_proc=args.n_proc)
    elif args.a == "sp500":
        save_sp500_tickers()
