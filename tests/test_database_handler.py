import os
from tests import import_source_modules
import_source_modules()

import datetime as dt
import pandas as pd
from pathlib import Path
from pandas.core.frame import DataFrame
from source import database_handler as dh
from source.database_handler import DataManager, download_all_data, \
    get_sp500_tickers, load_database, save_database
import pytest


def test_set_valid_data_path():
    new_path = str(Path.home())
    assert not dh.set_data_path(str(Path.home()))
    assert dh.path_to_data == Path(new_path).absolute()


def test_set_invalid_data_path():
    new_path = 'this_is_not_a_path'
    assert dh.set_data_path(new_path)
    os.removedirs(str(dh.path_to_data.absolute()))


def test_load_database_success():
    dh.set_data_path(Path(__file__).parent / 'test_data/')
    test_data = load_database('test_close.pkl')
    assert len(test_data) > 0
    assert isinstance(test_data, pd.DataFrame)
    del test_data


def test_load_database_fails_on_missing_file():
    dh.set_data_path(Path(__file__).parent / 'test_data/')
    with pytest.raises(Exception):
        assert load_database('not_a_file.pkl')


def test_load_database_fails_on_not_unpickleable():
    dh.set_data_path(Path(__file__).parent / 'test_data/')
    assert load_database('not_a_pickle_file.pkl') is None


def test_load_database_fails():
    dh.set_data_path(Path(__file__).parent / 'test_data/')
    with pytest.raises(Exception):
        assert load_database('not_a_file.pkl')


def test_save_database():
    dh.set_data_path(Path(__file__).parent / 'test_data/')
    test_data = load_database('test_close.pkl')
    
    assert save_database(test_data, 'test_save_close.pkl')
    assert (dh.path_to_data / 'test_save_close.pkl').exists()
    os.remove(str(dh.path_to_data / 'test_save_close.pkl'))


# TODO: Test for save_metadata


def test_create_database():
    _, goog_data, success = dh.create_database('GOOG')
    assert success
    assert isinstance(goog_data, pd.Series)
    assert goog_data.name == 'GOOG'
    assert goog_data.iloc[-1] > 0
    assert str(goog_data.index[0].date()) == '2004-08-19'


def test_create_database_with_start_end_dates():
    _, goog_data, success = dh.create_database('GOOG', '2020-01-02',
                                               '2020-11-13')
    assert success
    assert isinstance(goog_data, pd.Series)
    assert goog_data.name == 'GOOG'
    assert str(goog_data.index[0].date()) == '2020-01-02'
    assert str(goog_data.index[-1].date()) == '2020-11-12'


def test_create_database_with_start_date():
    _, goog_data, success = dh.create_database('GOOG', '2020-01-02')
    assert success
    assert isinstance(goog_data, pd.Series)
    assert goog_data.name == 'GOOG'
    assert str(goog_data.index[0].date()) == '2020-01-02'
    assert goog_data.index[-1].date() <= dt.datetime.now().date()


def test_create_database_with_start_date2():
    _, goog_data, success = dh.create_database('GOOG', '2020-01-02',
                                               '2020-01-03')
    assert success
    assert isinstance(goog_data, pd.Series)
    assert len(goog_data) == 1


def test_create_database_invalid_symbol():
    _, goog_data, success = dh.create_database('NOTASYMBOL')
    
    print(goog_data)
    assert not success
    assert goog_data is None


def test_add_stock():
    dh.set_data_path(Path(__file__).parent / 'test_data/')
    test_data = load_database('test_close.pkl')
    test_data = test_data[test_data.index <= dt.datetime(2020, 1, 1)]
    new_db, success = dh.add_stock(test_data, 'MSFT')
    assert success
    assert isinstance(new_db, pd.DataFrame)
    assert len(new_db.columns) == 4
    assert str(new_db.index[-1].date()) == '2019-12-31'


def test_add_stock_to_empty_dataframe():
    new_db, success = dh.add_stock(pd.DataFrame(), 'MSFT', end='2020-01-01')
    assert success
    assert isinstance(new_db, pd.DataFrame)
    assert len(new_db.columns) == 1
    assert str(new_db.index[-1].date()) == '2019-12-31'


# TODO: Test for returns
# TODO: test for dataframe updates


def test_create_database_from_scratch():
    test_path = Path(__file__)
    test_data_file_name = 'test_close.pkl'
    test_data_path = test_path.parent / 'test_data/'
    path_to_test_data = test_data_path / test_data_file_name
    dh.set_data_path(test_data_path)
    
    new_data = download_all_data(test_data_file_name,
                                 tickers=['GOOG', 'KR', 'TSLA'],
                                 n_proc=2)
    
    assert isinstance(new_data, pd.DataFrame)
    assert len(new_data.columns) == 3
    assert all(new_data.iloc[-1, :] > 0)
    assert path_to_test_data.exists()


def test_get_sp500_tickers():
    dh.set_data_path(Path(__file__).parent.parent / 'data/')
    sp500 = get_sp500_tickers()
    assert len(sp500) > 0
