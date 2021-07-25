from source.resources import OPERATION_BUY, OPERATION_SELL, Portfolio, Order,\
    Account
import pytest
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
    print(order)
    assert f"{order}" == "Buy A : 1 : 1.0000"
    order = Order('A', 1, 1.0001, OPERATION_BUY)
    assert f"{order}" == "Buy A : 1 : 1.0001"
    order = Order('A', 1, 1.00005, OPERATION_BUY)
    assert f"{order}" == "Buy A : 1 : 1.0001"
