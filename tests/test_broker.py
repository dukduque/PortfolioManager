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
from broker.base import BrokerHistory
from broker.paper import CostAwarePaperBroker, PaperBroker


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


# ---------------------------------------------------------------------------
# PaperBroker — cash tracking
# ---------------------------------------------------------------------------

def test_paper_broker_default_cash_is_zero():
    assert PaperBroker().get_cash() == 0.0


def test_paper_broker_initial_cash():
    assert PaperBroker(initial_cash=5_000.0).get_cash() == pytest.approx(5_000.0)


def test_paper_broker_buy_reduces_cash():
    broker = PaperBroker(initial_cash=10_000.0)
    broker.submit_order(Order('AAPL', 10, 100.0, OPERATION_BUY))
    assert broker.get_cash() == pytest.approx(10_000.0 - 10 * 100.0)


def test_paper_broker_sell_increases_cash():
    initial = Portfolio.create_from_vectors(['AAPL'], [10])
    broker = PaperBroker(initial_portfolio=initial, initial_cash=0.0)
    broker.submit_order(Order('AAPL', 5, 120.0, OPERATION_SELL))
    assert broker.get_cash() == pytest.approx(5 * 120.0)


# ---------------------------------------------------------------------------
# PaperBroker — deposit and withdraw
# ---------------------------------------------------------------------------

def test_paper_broker_deposit_increases_cash():
    broker = PaperBroker()
    broker.deposit(1_000.0)
    assert broker.get_cash() == pytest.approx(1_000.0)


def test_paper_broker_deposit_with_explicit_date():
    date = dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc)
    broker = PaperBroker()
    broker.deposit(2_500.0, date=date)
    assert broker.get_cash() == pytest.approx(2_500.0)


def test_paper_broker_withdraw_decreases_cash():
    broker = PaperBroker(initial_cash=5_000.0)
    broker.withdraw(1_000.0)
    assert broker.get_cash() == pytest.approx(4_000.0)


def test_paper_broker_withdraw_allows_negative_cash():
    """PaperBroker does not enforce a cash floor — intentional for simulation."""
    broker = PaperBroker(initial_cash=0.0)
    broker.withdraw(500.0)
    assert broker.get_cash() == pytest.approx(-500.0)


# ---------------------------------------------------------------------------
# PaperBroker — BrokerHistory: order history
# ---------------------------------------------------------------------------

def test_paper_broker_supports_cash_flow_history():
    assert PaperBroker().supports_cash_flow_history is True


def test_paper_broker_get_order_history_empty():
    broker = PaperBroker()
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)
    assert broker.get_order_history(start, end) == []


def test_paper_broker_get_order_history_after_buy():
    start = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2099, 12, 31, tzinfo=dt.timezone.utc)
    broker = PaperBroker(initial_cash=10_000.0)
    broker.submit_order(Order('AAPL', 5, 100.0, OPERATION_BUY))
    history = broker.get_order_history(start, end)
    assert len(history) == 1
    assert history[0].ticker == 'AAPL'
    assert history[0].qty == 5
    assert history[0].operation_type == OPERATION_BUY


def test_paper_broker_get_order_history_date_filter():
    broker = PaperBroker()
    t_before = dt.datetime(2024, 1,  1, tzinfo=dt.timezone.utc)
    t_inside = dt.datetime(2024, 3, 15, tzinfo=dt.timezone.utc)
    t_after  = dt.datetime(2024, 6,  1, tzinfo=dt.timezone.utc)
    start    = dt.datetime(2024, 2,  1, tzinfo=dt.timezone.utc)
    end      = dt.datetime(2024, 5,  1, tzinfo=dt.timezone.utc)
    broker._fill_history = [
        Fill('A', 1, 100.0, OPERATION_BUY,  t_before, 'id1'),
        Fill('B', 1, 100.0, OPERATION_BUY,  t_inside, 'id2'),
        Fill('C', 1, 100.0, OPERATION_SELL, t_after,  'id3'),
    ]
    result = broker.get_order_history(start, end)
    assert len(result) == 1
    assert result[0].ticker == 'B'


def test_paper_broker_multiple_orders_all_in_history():
    start = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2099, 12, 31, tzinfo=dt.timezone.utc)
    initial = Portfolio.create_from_vectors(['AAPL'], [20])
    broker = PaperBroker(initial_portfolio=initial, initial_cash=10_000.0)
    broker.submit_order(Order('AAPL', 5, 100.0, OPERATION_BUY))
    broker.submit_order(Order('AAPL', 3, 110.0, OPERATION_SELL))
    history = broker.get_order_history(start, end)
    assert len(history) == 2


# ---------------------------------------------------------------------------
# PaperBroker — BrokerHistory: cash flow history
# ---------------------------------------------------------------------------

def test_paper_broker_get_cash_flows_empty():
    broker = PaperBroker()
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)
    assert broker.get_cash_flows(start, end) == []


def test_paper_broker_get_cash_flows_deposit():
    date  = dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)
    broker = PaperBroker()
    broker.deposit(5_000.0, date=date)
    flows = broker.get_cash_flows(start, end)
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(5_000.0)
    assert flows[0].description == 'deposit'


def test_paper_broker_get_cash_flows_withdrawal():
    date  = dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc)
    start = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc)
    broker = PaperBroker(initial_cash=5_000.0)
    broker.withdraw(1_000.0, date=date)
    flows = broker.get_cash_flows(start, end)
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(-1_000.0)
    assert flows[0].description == 'withdrawal'


def test_paper_broker_get_cash_flows_date_filter():
    start = dt.datetime(2024, 2, 1, tzinfo=dt.timezone.utc)
    end   = dt.datetime(2024, 5, 1, tzinfo=dt.timezone.utc)
    broker = PaperBroker()
    broker.deposit(1_000.0, date=dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc))  # before range
    broker.deposit(2_000.0, date=dt.datetime(2024, 3, 1, tzinfo=dt.timezone.utc))  # in range
    broker.deposit(3_000.0, date=dt.datetime(2024, 6, 1, tzinfo=dt.timezone.utc))  # after range
    flows = broker.get_cash_flows(start, end)
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(2_000.0)


# ---------------------------------------------------------------------------
# CostAwarePaperBroker — cash tracking with spread
# ---------------------------------------------------------------------------

def test_cost_aware_buy_reduces_cash_by_adjusted_price():
    broker = CostAwarePaperBroker(cost_bps=10.0, initial_cash=10_000.0)
    broker.submit_order(Order('AA', 10, 100.0, OPERATION_BUY))
    expected = 10_000.0 - 10 * 100.0 * (1 + 10 / 10_000)
    assert broker.get_cash() == pytest.approx(expected)


def test_cost_aware_sell_increases_cash_by_adjusted_price():
    initial = Portfolio.create_from_vectors(['AA'], [20])
    broker = CostAwarePaperBroker(cost_bps=10.0, initial_portfolio=initial, initial_cash=0.0)
    broker.submit_order(Order('AA', 10, 100.0, OPERATION_SELL))
    expected = 10 * 100.0 * (1 - 10 / 10_000)
    assert broker.get_cash() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# BrokerHistory base — NotImplementedError for cash flows
# ---------------------------------------------------------------------------

def test_broker_history_base_get_cash_flows_raises():
    """BrokerHistory.get_cash_flows() raises NotImplementedError by default."""
    class _MinimalBroker(BrokerHistory):
        def get_order_history(self, start, end):
            return []

    broker = _MinimalBroker()
    assert broker.supports_cash_flow_history is False
    with pytest.raises(NotImplementedError):
        broker.get_cash_flows(
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 12, 31),
        )
