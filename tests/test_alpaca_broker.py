"""Tests for AlpacaBroker — all Alpaca API calls are mocked via unittest.mock."""
from tests import import_source_modules
import_source_modules()

import datetime as dt
import pytest
from unittest.mock import MagicMock, patch

# Skip the whole module if alpaca-py is not installed.
pytest.importorskip("alpaca.trading.client", reason="alpaca-py not installed")

from alpaca.trading.enums import OrderSide, OrderStatus
from broker.alpaca import AlpacaBroker, _to_utc
from resources import Fill, Order, Portfolio, OPERATION_BUY, OPERATION_SELL

UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_broker():
    """Return (AlpacaBroker, mock_client) with TradingClient replaced by a mock."""
    mock_client = MagicMock()
    with patch("broker.alpaca.TradingClient", return_value=mock_client):
        broker = AlpacaBroker("fake-key", "fake-secret", paper=True)
    # broker._client is now mock_client; subsequent calls work without the patch.
    return broker, mock_client


def _alpaca_order(
    status=None,
    side=None,
    symbol="AAPL",
    filled_qty="10",
    filled_avg_price="150.00",
    filled_at=None,
    order_id="order-abc",
):
    """Build a mock Alpaca order object with sensible defaults."""
    o = MagicMock()
    o.status       = status or OrderStatus.FILLED
    o.side         = side or OrderSide.BUY
    o.symbol       = symbol
    o.filled_qty   = filled_qty
    o.filled_avg_price = filled_avg_price
    o.filled_at    = filled_at or dt.datetime(2024, 1, 2, 10, tzinfo=UTC)
    o.id           = order_id
    return o


def _alpaca_activity(
    date=None,
    transaction_time=None,
    net_amount="1000.00",
    description="Wire transfer",
    activity_type="JNLC",
):
    """Build a mock Alpaca account activity object."""
    a = MagicMock()
    a.date             = date
    a.transaction_time = transaction_time
    a.net_amount       = net_amount
    a.description      = description
    a.activity_type    = activity_type
    return a


# ---------------------------------------------------------------------------
# BrokerBase — submit_order
# ---------------------------------------------------------------------------

def test_submit_order_buy_returns_broker_id():
    broker, client = _make_broker()
    client.submit_order.return_value = MagicMock(id="alpaca-123")
    order_id = broker.submit_order(Order("AAPL", 10, 0.0, OPERATION_BUY))
    assert order_id == "alpaca-123"
    client.submit_order.assert_called_once()


def test_submit_order_sell_uses_sell_side():
    broker, client = _make_broker()
    client.submit_order.return_value = MagicMock(id="alpaca-456")
    broker.submit_order(Order("AAPL", 5, 0.0, OPERATION_SELL))
    request = client.submit_order.call_args[0][0]
    assert request.side == OrderSide.SELL


def test_submit_order_buy_uses_buy_side():
    broker, client = _make_broker()
    client.submit_order.return_value = MagicMock(id="alpaca-789")
    broker.submit_order(Order("MSFT", 3, 0.0, OPERATION_BUY))
    request = client.submit_order.call_args[0][0]
    assert request.side == OrderSide.BUY


# ---------------------------------------------------------------------------
# BrokerBase — get_fill
# ---------------------------------------------------------------------------

def test_get_fill_returns_fill_when_filled():
    broker, client = _make_broker()
    client.get_order_by_id.return_value = _alpaca_order(
        symbol="MSFT", filled_qty="7", filled_avg_price="300.50"
    )
    fill = broker.get_fill("order-abc")
    assert fill is not None
    assert fill.ticker == "MSFT"
    assert fill.qty == pytest.approx(7.0)
    assert fill.fill_price == pytest.approx(300.50)
    assert fill.operation_type == OPERATION_BUY


def test_get_fill_returns_none_when_pending():
    broker, client = _make_broker()
    client.get_order_by_id.return_value = _alpaca_order(status=OrderStatus.PENDING_NEW)
    assert broker.get_fill("some-id") is None


def test_get_fill_maps_sell_side_correctly():
    broker, client = _make_broker()
    client.get_order_by_id.return_value = _alpaca_order(side=OrderSide.SELL)
    fill = broker.get_fill("order-abc")
    assert fill.operation_type == OPERATION_SELL


def test_get_fill_broker_order_id_matches():
    broker, client = _make_broker()
    client.get_order_by_id.return_value = _alpaca_order(order_id="the-id")
    fill = broker.get_fill("the-id")
    assert fill.broker_order_id == "the-id"


# ---------------------------------------------------------------------------
# BrokerBase — get_positions
# ---------------------------------------------------------------------------

def test_get_positions_maps_to_portfolio():
    broker, client = _make_broker()
    client.get_all_positions.return_value = [
        MagicMock(symbol="AAPL", qty="10"),
        MagicMock(symbol="GOOG", qty="3"),
    ]
    portfolio = broker.get_positions()
    assert portfolio.get_position("AAPL") == pytest.approx(10.0)
    assert portfolio.get_position("GOOG") == pytest.approx(3.0)


def test_get_positions_excludes_zero_qty():
    broker, client = _make_broker()
    client.get_all_positions.return_value = [MagicMock(symbol="AAPL", qty="0")]
    portfolio = broker.get_positions()
    assert "AAPL" not in portfolio.assets


def test_get_positions_empty_when_no_positions():
    broker, client = _make_broker()
    client.get_all_positions.return_value = []
    portfolio = broker.get_positions()
    assert len(portfolio.assets) == 0


# ---------------------------------------------------------------------------
# BrokerBase — get_cash
# ---------------------------------------------------------------------------

def test_get_cash_returns_float():
    broker, client = _make_broker()
    client.get_account.return_value = MagicMock(cash="12345.67")
    assert broker.get_cash() == pytest.approx(12345.67)


def test_get_cash_zero():
    broker, client = _make_broker()
    client.get_account.return_value = MagicMock(cash="0.00")
    assert broker.get_cash() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# BrokerBase — cancel_order
# ---------------------------------------------------------------------------

def test_cancel_order_returns_true_on_success():
    broker, client = _make_broker()
    client.cancel_order_by_id.return_value = None
    assert broker.cancel_order("order-xyz") is True


def test_cancel_order_returns_false_on_exception():
    broker, client = _make_broker()
    client.cancel_order_by_id.side_effect = Exception("not found")
    assert broker.cancel_order("order-xyz") is False


# ---------------------------------------------------------------------------
# BrokerBase — is_market_open
# ---------------------------------------------------------------------------

def test_is_market_open_true():
    broker, client = _make_broker()
    client.get_clock.return_value = MagicMock(is_open=True)
    assert broker.is_market_open() is True


def test_is_market_open_false():
    broker, client = _make_broker()
    client.get_clock.return_value = MagicMock(is_open=False)
    assert broker.is_market_open() is False


# ---------------------------------------------------------------------------
# BrokerHistory — supports_cash_flow_history
# ---------------------------------------------------------------------------

def test_supports_cash_flow_history_is_true():
    broker, _ = _make_broker()
    assert broker.supports_cash_flow_history is True


# ---------------------------------------------------------------------------
# BrokerHistory — get_order_history
# ---------------------------------------------------------------------------

def test_get_order_history_returns_filled_orders():
    broker, client = _make_broker()
    client.get_orders.return_value = [
        _alpaca_order(symbol="AAPL", filled_qty="5",  filled_avg_price="200.00"),
        _alpaca_order(symbol="GOOG", filled_qty="2",  filled_avg_price="100.00",
                      side=OrderSide.SELL),
    ]
    fills = broker.get_order_history(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert len(fills) == 2
    assert fills[0].ticker == "AAPL"
    assert fills[0].qty == pytest.approx(5.0)
    assert fills[0].operation_type == OPERATION_BUY
    assert fills[1].operation_type == OPERATION_SELL


def test_get_order_history_excludes_non_filled():
    broker, client = _make_broker()
    client.get_orders.return_value = [
        _alpaca_order(symbol="AAPL"),                # FILLED — included
        _alpaca_order(status=OrderStatus.CANCELED),  # CANCELED — excluded
        _alpaca_order(status=OrderStatus.EXPIRED),   # EXPIRED — excluded
    ]
    fills = broker.get_order_history(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert len(fills) == 1
    assert fills[0].ticker == "AAPL"


def test_get_order_history_empty_when_no_orders():
    broker, client = _make_broker()
    client.get_orders.return_value = []
    fills = broker.get_order_history(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert fills == []


def test_get_order_history_raises_on_api_error():
    broker, client = _make_broker()
    client.get_orders.side_effect = Exception("network error")
    with pytest.raises(Exception, match="network error"):
        broker.get_order_history(
            dt.datetime(2024, 1, 1, tzinfo=UTC),
            dt.datetime(2024, 12, 31, tzinfo=UTC),
        )


def test_get_order_history_broker_id_preserved():
    broker, client = _make_broker()
    client.get_orders.return_value = [_alpaca_order(order_id="order-xyz")]
    fills = broker.get_order_history(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert fills[0].broker_order_id == "order-xyz"


# ---------------------------------------------------------------------------
# BrokerHistory — get_cash_flows
# ---------------------------------------------------------------------------

def test_get_cash_flows_within_range():
    broker, client = _make_broker()
    client.get_account_activities.return_value = [
        _alpaca_activity(date=dt.date(2024, 3, 15), net_amount="5000.00",
                         description="Wire deposit"),
    ]
    flows = broker.get_cash_flows(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(5000.0)
    assert flows[0].description == "Wire deposit"


def test_get_cash_flows_excludes_outside_range():
    broker, client = _make_broker()
    client.get_account_activities.return_value = [
        _alpaca_activity(date=dt.date(2023, 6, 1), net_amount="1000.00"),  # before start
    ]
    flows = broker.get_cash_flows(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert flows == []


def test_get_cash_flows_falls_back_to_transaction_time():
    """Uses transaction_time when date attribute is None."""
    broker, client = _make_broker()
    client.get_account_activities.return_value = [
        _alpaca_activity(
            date=None,
            transaction_time=dt.datetime(2024, 4, 1, tzinfo=UTC),
            net_amount="750.00",
        ),
    ]
    flows = broker.get_cash_flows(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert len(flows) == 1
    assert flows[0].amount == pytest.approx(750.0)


def test_get_cash_flows_skips_activity_with_no_date():
    """Activities with neither date nor transaction_time are skipped."""
    broker, client = _make_broker()
    client.get_account_activities.return_value = [
        _alpaca_activity(date=None, transaction_time=None),
    ]
    flows = broker.get_cash_flows(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert flows == []


def test_get_cash_flows_multiple_activities():
    broker, client = _make_broker()
    client.get_account_activities.return_value = [
        _alpaca_activity(date=dt.date(2024, 2, 1), net_amount="10000.00"),
        _alpaca_activity(date=dt.date(2024, 4, 1), net_amount="-2000.00"),
    ]
    flows = broker.get_cash_flows(
        dt.datetime(2024, 1, 1, tzinfo=UTC),
        dt.datetime(2024, 12, 31, tzinfo=UTC),
    )
    assert len(flows) == 2
    amounts = sorted(f.amount for f in flows)
    assert amounts[0] == pytest.approx(-2000.0)
    assert amounts[1] == pytest.approx(10000.0)


def test_get_cash_flows_raises_on_api_error():
    broker, client = _make_broker()
    client.get_account_activities.side_effect = Exception("timeout")
    with pytest.raises(Exception, match="timeout"):
        broker.get_cash_flows(
            dt.datetime(2024, 1, 1, tzinfo=UTC),
            dt.datetime(2024, 12, 31, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# _to_utc helper
# ---------------------------------------------------------------------------

def test_to_utc_from_date_object():
    result = _to_utc(dt.date(2024, 6, 15))
    assert isinstance(result, dt.datetime)
    assert result.tzinfo is not None
    assert result.year == 2024 and result.month == 6 and result.day == 15


def test_to_utc_from_naive_datetime():
    result = _to_utc(dt.datetime(2024, 6, 15, 10, 30))
    assert result.tzinfo == UTC


def test_to_utc_from_aware_datetime_is_unchanged():
    aware = dt.datetime(2024, 6, 15, 10, 30, tzinfo=UTC)
    assert _to_utc(aware) == aware


def test_to_utc_from_pandas_timestamp():
    import pandas as pd
    ts = pd.Timestamp("2024-06-15 10:30:00")
    result = _to_utc(ts)
    assert isinstance(result, dt.datetime)
    assert result.tzinfo is not None
    assert result.year == 2024 and result.month == 6 and result.day == 15
