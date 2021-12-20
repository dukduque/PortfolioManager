from tests import import_source_modules
import_source_modules()

import datetime as dt
from pathlib import Path
from resources import OPERATION_BUY, OPERATION_SELL, Portfolio, Order,\
    Account, create_new_account, load_account, set_account_path
'''
=============================================
Portfolio class tests
=============================================
'''


def test_create_empty():
    portfolio = Portfolio.create_empty()
    assert len(portfolio._data) == 0
    assert portfolio.get_position('A') == 0


def test_create_from_vectors():
    portfolio = Portfolio.create_from_vectors(['A', 'B'], [1, 2])
    assert len(portfolio._data) == 2
    assert portfolio.get_position('A') == 1
    assert portfolio.get_position('B') == 2
    assert portfolio.assets == set(['A', 'B'])


def test_create_from_transactions():
    initial_portfolio = Portfolio.create_from_vectors(['A', 'B'], [1, 2])
    transactions = [
        Order('C', 10, 1.5, OPERATION_BUY),
        Order('D', 5, 0.5, OPERATION_BUY),
    ]
    new_portfolio = Portfolio.create_from_transaction(initial_portfolio,
                                                      transactions)
    assert len(new_portfolio._data) == 4
    assert new_portfolio.assets == set(['A', 'B', 'C', 'D'])
    assert new_portfolio.get_position('A') == 1
    assert new_portfolio.get_position('B') == 2
    assert new_portfolio.get_position('C') == 10
    assert new_portfolio.get_position('D') == 5
    
    transactions = [
        Order('A', 1, 1.5, OPERATION_SELL),
        Order('D', 5, 0.5, OPERATION_SELL),
    ]
    new_portfolio = Portfolio.create_from_transaction(new_portfolio,
                                                      transactions)
    assert len(new_portfolio._data) == 2
    assert new_portfolio.assets == set(['B', 'C'])
    assert new_portfolio.get_position('A') == 0
    assert new_portfolio.get_position('B') == 2
    assert new_portfolio.get_position('C') == 10
    assert new_portfolio.get_position('D') == 0
    
    # Selling more than the current position
    new_portfolio = Portfolio.create_from_transaction(
        new_portfolio, [Order('B', 3, 0.1, OPERATION_SELL)])
    assert new_portfolio is None
    
    # Selling a stock that is not in the portfolio
    new_portfolio = Portfolio.create_from_transaction(
        Portfolio.create_empty(), [Order('A', 3, 0.1, OPERATION_SELL)])
    assert new_portfolio is None


'''
=============================================
Order class tests
=============================================
'''


def test_order_str():
    order = Order('A', 1, 1.0, OPERATION_BUY)
    assert f"{order}" == "Buy A : 1 : 1.0000"
    order = Order('A', 1, 1.0001, OPERATION_BUY)
    assert f"{order}" == "Buy A : 1 : 1.0001"
    order = Order('A', 1, 1.00005, OPERATION_BUY)
    assert f"{order}" == "Buy A : 1 : 1.0001"


'''
=============================================
Account class tests
=============================================
'''


def test_new_account():
    account = Account('holder', dt.datetime.now())
    assert account.last_transaction <= dt.datetime.now()
    assert len(account.portfolio.assets) == 0


def test_account_deposit():
    account = Account('holder', dt.datetime.now())
    deposit_date = dt.datetime(2021, 12, 12, 10, 10)
    assert account.deposit(deposit_date, 1_000)
    assert account.cash_onhand == 1_000


def test_account_withdraw():
    account = Account('holder', dt.datetime.now())
    deposit_date = dt.datetime(2021, 12, 12, 10, 10)
    account.deposit(deposit_date, 1_000)
    withdraw_date = dt.datetime(2021, 12, 12, 10, 14)
    
    assert not account.withdraw(withdraw_date, 10_000)
    assert account.cash_onhand == 1_000
    
    assert account.withdraw(withdraw_date, 100)
    assert account.cash_onhand == 900


def test_account_update():
    account = Account('holder', dt.datetime.now())
    account.deposit(dt.datetime(2021, 12, 12, 10, 10), 1_000)
    
    update_status = account.update_account(dt.datetime(2021, 12, 20, 9, 35),
                                           orders=[
                                               Order('ABC', 100, 1.2,
                                                     OPERATION_BUY),
                                               Order('XYZ', 1, 453.2,
                                                     OPERATION_BUY)
                                           ])
    
    assert update_status
    assert account.portfolio.get_position('ABC') == 100
    assert account.portfolio.get_position('XYZ') == 1
    assert account.cash_onhand == 1_000 - 1.2 * 100 - 453.2


def test_account_update_sell():
    account = Account('holder', dt.datetime.now())
    account.deposit(dt.datetime(2021, 12, 12, 10, 10), 1_000)
    account.update_account(dt.datetime(2021, 12, 20, 9, 30),
                           orders=[
                               Order('ABC', 100, 1.2, OPERATION_BUY),
                               Order('XYZ', 1000, 0.2, OPERATION_BUY)
                           ])
    initial_cash_onhand = account.cash_onhand
    
    update_status = account.update_account(dt.datetime(2021, 12, 21, 9, 30),
                                           orders=[
                                               Order('ABC', 10, 2.2,
                                                     OPERATION_BUY),
                                               Order('XYZ', 10, 1.0,
                                                     OPERATION_SELL)
                                           ])
    assert update_status
    assert account.portfolio.get_position('ABC') == 110
    assert account.portfolio.get_position('XYZ') == 990
    assert account.cash_onhand == initial_cash_onhand - 10 * 2.2 + 10 * 1.0


def test_account_update_sells_missing_asset():
    account = Account('holder', dt.datetime.now())
    account.deposit(dt.datetime(2021, 12, 12, 10, 10), 1_000)
    account.update_account(dt.datetime(2021, 12, 20, 9, 30),
                           orders=[Order('ABC', 100, 1.2, OPERATION_BUY)])
    initial_cash_onhand = account.cash_onhand
    
    update_status = account.update_account(dt.datetime(2021, 12, 21, 9, 30),
                                           orders=[
                                               Order('ABC', 10, 2.2,
                                                     OPERATION_BUY),
                                               Order('XYZ', 10, 2.2,
                                                     OPERATION_SELL)
                                           ])
    assert not update_status
    assert account.cash_onhand == initial_cash_onhand
    assert account.portfolio.get_position('ABC') == 100
    assert account.portfolio.get_position('XYZ') == 0


def test_load_account():
    set_account_path(Path(__file__).parent / 'test_data/')
    create_new_account('holder_a', dt.datetime.now)
    
    account = load_account('holder_a')
    assert account.holder == 'holder_a'
    
    account = load_account('holder_b')
    assert account is None
