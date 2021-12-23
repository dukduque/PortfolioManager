from tests import import_source_modules
import_source_modules()

import datetime as dt
import pandas as pd
from pathlib import Path
from resources import OPERATION_BUY, OPERATION_SELL, Portfolio, Order,\
    Account, build_account_history, create_new_account, \
    generate_orders, load_account, set_account_path
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


def test_create_from_vectors_is_none_for_negative_qty():
    portfolio = Portfolio.create_from_vectors(['A', 'B'], [1, -2])
    assert portfolio is None


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


def test_get_postion():
    initial_portfolio = Portfolio.create_from_vectors(['A', 'B'], [1, 2])
    assert initial_portfolio.get_position('A') == 1
    assert initial_portfolio.get_position('B') == 2
    assert initial_portfolio.get_position('C') == 0


def test_fractional_postions():
    initial_portfolio = Portfolio.create_from_vectors(['A', 'B'], [1, 2.5])
    assert not initial_portfolio.position_is_fractional('A')
    assert initial_portfolio.position_is_fractional('B')
    assert not initial_portfolio.position_is_fractional('C')
    print(initial_portfolio)


def test_modify_position():
    initial_portfolio = Portfolio.create_from_vectors(['A', 'B'], [1, 2.5])
    initial_portfolio.modify_position('A', 12)
    initial_portfolio.modify_position('C', 1)
    assert initial_portfolio.get_position('A') == 12
    assert initial_portfolio.get_position('B') == 2.5
    assert initial_portfolio.get_position('C') == 1


def test_add_portfolios():
    portfolio_a = Portfolio.create_from_vectors(['A', 'B'], [1, 2.5])
    portfolio_b = Portfolio.create_from_vectors(['A', 'C'], [1, 3])
    portfolio_c = Portfolio.create_from_vectors(['C'], [10])
    final_portfolio = (portfolio_a + portfolio_b) + portfolio_c
    assert final_portfolio.get_position('A') == 2
    assert final_portfolio.get_position('B') == 2.5
    assert final_portfolio.get_position('C') == 13


def test_add_none_portfolios():
    portfolio_a = Portfolio.create_from_vectors(['A', 'B'], [0, 2.5])
    assert portfolio_a + None is None


def test_add_invalid_portfolios():
    portfolio_a = Portfolio.create_from_vectors(['A', 'B'], [0, 2.5])
    portfolio_a._data['A']['qty'] = -1
    portfolio_b = Portfolio.create_from_vectors(['A', 'B'], [1, 2.5])
    assert portfolio_a + portfolio_b is None


def test_str_repr():
    portfolio_a = Portfolio.create_from_vectors(['A', 'B'], [0, 2.5])
    assert str(portfolio_a) == f'{"A":8s}{0}\n{"B":8s}{2.5}\n'
    assert portfolio_a.__str__() == portfolio_a.__repr__()


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
    assert order.__str__() == order.__repr__()


'''
=============================================
Account class tests
=============================================
'''


def test_new_account():
    account = Account('holder', dt.datetime(2021, 10, 31, 9, 30))
    assert account.last_transaction <= dt.datetime(2021, 10, 31, 9, 30)
    assert len(account.portfolio.assets) == 0


def test_account_deposit():
    account = Account('holder', dt.datetime(2021, 10, 31, 9, 30))
    deposit_date = dt.datetime(2021, 12, 12, 10, 10)
    assert account.deposit(deposit_date, 1_000)
    assert account.cash_onhand == 1_000


def test_account_withdraw():
    account = Account('holder', dt.datetime(2021, 10, 31, 9, 30))
    deposit_date = dt.datetime(2021, 12, 12, 10, 10)
    account.deposit(deposit_date, 1_000)
    withdraw_date = dt.datetime(2021, 12, 12, 10, 14)
    
    assert not account.withdraw(withdraw_date, 10_000)
    assert account.cash_onhand == 1_000
    
    assert account.withdraw(withdraw_date, 100)
    assert account.cash_onhand == 900


def test_account_update():
    account = Account('holder', dt.datetime(2021, 10, 31, 9, 30))
    account.deposit(dt.datetime(2021, 12, 13, 10, 10), 1_000)
    
    update_status = account.update_account(dt.datetime(2021, 12, 17, 9, 35),
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
    account = Account('holder', dt.datetime(2021, 10, 31, 9, 30))
    account.deposit(dt.datetime(2021, 12, 12, 10, 10), 1_000)
    account.update_account(dt.datetime(2021, 12, 13, 9, 30),
                           orders=[
                               Order('ABC', 100, 1.2, OPERATION_BUY),
                               Order('XYZ', 1000, 0.2, OPERATION_BUY)
                           ])
    initial_cash_onhand = account.cash_onhand
    
    update_status = account.update_account(dt.datetime(2021, 12, 17, 9, 30),
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
    account = Account('holder', dt.datetime(2021, 10, 31, 9, 30))
    account.deposit(dt.datetime(2021, 12, 12, 10, 10), 1_000)
    account.update_account(dt.datetime(2021, 12, 13, 9, 30),
                           orders=[Order('ABC', 100, 1.2, OPERATION_BUY)])
    initial_cash_onhand = account.cash_onhand
    
    update_status = account.update_account(dt.datetime(2021, 12, 17, 9, 30),
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


def test_account_operations_history():
    set_account_path(Path(__file__).parent / 'test_data/')
    account = create_new_account('holder_a', dt.datetime(2021, 12, 12))
    account.update_account(dt.datetime(2021, 12, 13, 9, 30),
                           orders=[Order('ABC', 1, 123, OPERATION_BUY)])
    account.update_account(dt.datetime(2021, 12, 15, 9, 30),
                           orders=[
                               Order('ABC', 0.5, 123, OPERATION_SELL),
                               Order('XYZ', 2, 456, OPERATION_BUY),
                           ])
    account.update_account(dt.datetime(2021, 12, 16, 9, 30),
                           orders=[Order('XYZ', 1, 450, OPERATION_SELL)])
    
    history = account.operations_history()
    assert len(history) == 4
    assert history[0][1] == 123
    assert history[1][1] == 2 * 456
    assert history[2][1] == -0.5 * 123
    assert history[3][1] == -450
    
    account = load_account('holder_a')
    assert account.holder == 'holder_a'
    
    account = load_account('holder_b')
    assert account is None


def test_load_account():
    set_account_path(Path(__file__).parent / 'test_data/')
    create_new_account('holder_a', dt.datetime.now)
    
    account = load_account('holder_a')
    assert account.holder == 'holder_a'
    
    account = load_account('holder_b')
    assert account is None


def test_generate_orders():
    old_portfolio = Portfolio.create_from_vectors(['A', 'B', 'D'], [1, 2, 4])
    new_portfolio = Portfolio.create_from_vectors(['A', 'C', 'D'], [10, 2, 4])
    
    orders = generate_orders(old_portfolio,
                             new_portfolio,
                             prices={
                                 "A": 10.3,
                                 "B": 31.4,
                                 "C": 0.341,
                                 "D": 1.7
                             })
    assert len(orders) == 3
    
    assert orders[0].ticker == "A"
    assert orders[0].qty == 9
    assert orders[0].price == 10.3
    assert orders[0].operation_type == OPERATION_BUY
    
    assert orders[1].ticker == "B"
    assert orders[1].qty == 2
    assert orders[1].price == 31.4
    assert orders[1].operation_type == OPERATION_SELL
    
    assert orders[2].ticker == "C"
    assert orders[2].qty == 2
    assert orders[2].price == 0.341
    assert orders[2].operation_type == OPERATION_BUY


def test_generate_orders_missing_prices():
    old_portfolio = Portfolio.create_from_vectors(['A', 'B', 'D'], [1, 2, 4])
    new_portfolio = Portfolio.create_from_vectors(['A', 'C', 'D'], [10, 2, 3])
    
    orders = generate_orders(old_portfolio,
                             new_portfolio,
                             prices={
                                 "A": 10.3,
                                 "B": 31.4,
                                 "C": 0.341
                             })
    # Asset D is missing from the price dictionary
    assert orders is None


def test_build_history():
    class DataManagerMock:
        def get_prices(_, assets):
            df = pd.DataFrame(data={
                'A': [1, 1, 1, 10, 11],
                'B': [3, 4, 5, 2, 3],
                'C': [100, 100, 110, 100, 100],
            },
                              index=[
                                  dt.datetime(2021, 12, 13),
                                  dt.datetime(2021, 12, 14),
                                  dt.datetime(2021, 12, 15),
                                  dt.datetime(2021, 12, 16),
                                  dt.datetime(2021, 12, 17),
                              ])
            return df[assets]
    
    data_mocker = DataManagerMock()
    portfolios = {
        dt.datetime(2021, 12, 13):
        Portfolio.create_from_vectors(['A'], [10]),
        dt.datetime(2021, 12, 15):
        Portfolio.create_from_vectors(['A', 'C'], [10, 1]),
    }
    history_dates, account_history = build_account_history(
        portfolios, data_mocker)
    
    assert history_dates[0].day == 13
    assert history_dates[1].day == 14
    assert history_dates[2].day == 15
    assert history_dates[3].day == 15
    assert history_dates[4].day == 16
    
    assert account_history == [10, 10, 10, 10 + 110, 100 + 100, 110 + 100]
