from tests import import_source_modules
import_source_modules()

import datetime as dt
import pytest

from resources import (
    Account,
    Fill,
    Order,
    Portfolio,
    OPERATION_BUY,
    OPERATION_SELL,
)
from broker.paper import PaperBroker


# ---------------------------------------------------------------------------
# PaperBroker — order submission and fills
# ---------------------------------------------------------------------------

def test_paper_broker_submit_buy_returns_order_id():
    broker = PaperBroker()
    order_id = broker.submit_order(Order('AAPL', 10, 150.0, OPERATION_BUY))
    assert order_id is not None
    assert len(order_id) > 0


def test_paper_broker_fill_matches_order():
    broker = PaperBroker()
    order = Order('AAPL', 10, 150.0, OPERATION_BUY)
    order_id = broker.submit_order(order)
    fill = broker.get_fill(order_id)
    assert fill is not None
    assert fill.ticker == 'AAPL'
    assert fill.qty == 10
    assert fill.fill_price == 150.0
    assert fill.operation_type == OPERATION_BUY
    assert fill.broker_order_id == order_id
    assert isinstance(fill.fill_time, dt.datetime)


def test_paper_broker_sell_fill():
    initial = Portfolio.create_from_vectors(['AAPL'], [10])
    broker = PaperBroker(initial_portfolio=initial)
    order_id = broker.submit_order(Order('AAPL', 5, 155.0, OPERATION_SELL))
    fill = broker.get_fill(order_id)
    assert fill.ticker == 'AAPL'
    assert fill.qty == 5
    assert fill.fill_price == 155.0
    assert fill.operation_type == OPERATION_SELL


def test_paper_broker_unknown_order_id_returns_none():
    broker = PaperBroker()
    assert broker.get_fill('nonexistent-id') is None


# ---------------------------------------------------------------------------
# PaperBroker — position tracking
# ---------------------------------------------------------------------------

def test_paper_broker_positions_after_buy():
    broker = PaperBroker()
    broker.submit_order(Order('AAPL', 10, 150.0, OPERATION_BUY))
    broker.submit_order(Order('GOOG', 2, 2800.0, OPERATION_BUY))
    positions = broker.get_positions()
    assert positions.get_position('AAPL') == 10
    assert positions.get_position('GOOG') == 2


def test_paper_broker_positions_after_sell():
    initial = Portfolio.create_from_vectors(['AAPL'], [10])
    broker = PaperBroker(initial_portfolio=initial)
    broker.submit_order(Order('AAPL', 10, 155.0, OPERATION_SELL))
    positions = broker.get_positions()
    assert positions.get_position('AAPL') == 0


def test_paper_broker_positions_accumulate_over_multiple_buys():
    broker = PaperBroker()
    broker.submit_order(Order('AAPL', 10, 150.0, OPERATION_BUY))
    broker.submit_order(Order('AAPL', 5, 152.0, OPERATION_BUY))
    assert broker.get_positions().get_position('AAPL') == 15


def test_paper_broker_oversell_raises():
    initial = Portfolio.create_from_vectors(['AAPL'], [5])
    broker = PaperBroker(initial_portfolio=initial)
    with pytest.raises(ValueError):
        broker.submit_order(Order('AAPL', 10, 150.0, OPERATION_SELL))


# ---------------------------------------------------------------------------
# PaperBroker — market state and cancel
# ---------------------------------------------------------------------------

def test_paper_broker_market_open_by_default():
    broker = PaperBroker()
    assert broker.is_market_open() is True


def test_paper_broker_set_market_open():
    broker = PaperBroker()
    broker.set_market_open(False)
    assert broker.is_market_open() is False
    broker.set_market_open(True)
    assert broker.is_market_open() is True


def test_paper_broker_cancel_returns_false():
    # PaperBroker fills immediately, so cancel is always a no-op.
    broker = PaperBroker()
    order_id = broker.submit_order(Order('AAPL', 10, 150.0, OPERATION_BUY))
    assert broker.cancel_order(order_id) is False


# ---------------------------------------------------------------------------
# PaperBroker — submit_and_await_fill (inherited from BrokerBase)
# ---------------------------------------------------------------------------

def test_submit_and_await_fill_returns_fill():
    broker = PaperBroker()
    fill = broker.submit_and_await_fill(
        Order('MSFT', 5, 300.0, OPERATION_BUY), timeout_seconds=5
    )
    assert fill is not None
    assert fill.ticker == 'MSFT'
    assert fill.qty == 5


# ---------------------------------------------------------------------------
# Account.update_account_from_fills
# ---------------------------------------------------------------------------

def test_account_update_from_fills_updates_portfolio():
    account = Account('holder', dt.datetime(2024, 1, 1))
    account.deposit(dt.datetime(2024, 1, 1), 10_000.0)

    broker = PaperBroker()
    order_id = broker.submit_order(Order('AAPL', 10, 150.0, OPERATION_BUY))
    fill = broker.get_fill(order_id)

    account.update_account_from_fills(dt.datetime(2024, 1, 2), [fill])

    assert account.portfolio.get_position('AAPL') == 10
    assert account.cash_onhand == pytest.approx(10_000.0 - 10 * 150.0)


def test_account_fill_price_differs_from_optimizer_price():
    """Actual fill price (broker) may differ from optimizer's estimated price."""
    account = Account('holder', dt.datetime(2024, 1, 1))
    account.deposit(dt.datetime(2024, 1, 1), 10_000.0)

    fill = Fill(
        ticker='AAPL',
        qty=10,
        fill_price=151.50,       # broker charged 151.50
        operation_type=OPERATION_BUY,
        fill_time=dt.datetime(2024, 1, 2, 10, 0),
        broker_order_id='broker-123',
    )

    account.update_account_from_fills(dt.datetime(2024, 1, 2), [fill])

    assert account.portfolio.get_position('AAPL') == 10
    # Cash reflects broker fill price, not a hypothetical optimizer price.
    assert account.cash_onhand == pytest.approx(10_000.0 - 10 * 151.50)


# ---------------------------------------------------------------------------
# Fill dataclass
# ---------------------------------------------------------------------------

def test_fill_dataclass_fields():
    fill = Fill(
        ticker='TSLA',
        qty=3.5,
        fill_price=200.0,
        operation_type=OPERATION_BUY,
        fill_time=dt.datetime(2024, 1, 2, 10, 30),
        broker_order_id='abc-123',
    )
    assert fill.ticker == 'TSLA'
    assert fill.qty == 3.5
    assert fill.fill_price == 200.0
    assert fill.operation_type == OPERATION_BUY
    assert fill.broker_order_id == 'abc-123'
