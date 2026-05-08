"""Alpaca broker integration using the alpaca-py SDK.

Install the dependency with:
    pip install alpaca-py

API keys are read from constructor arguments. Load them from environment
variables or a secrets manager — never hard-code them:

    import os
    broker = AlpacaBroker(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        paper=True,  # set False for live trading
    )
"""
import datetime as dt
import logging
from typing import List, Optional

from broker.base import BrokerBase, BrokerHistory
from resources import CashFlow, Fill, Order, Portfolio, OPERATION_BUY, OPERATION_SELL

log = logging.getLogger(__name__)

# Activity types that represent cash deposits and withdrawals on Alpaca.
_CASH_FLOW_ACTIVITY_TYPES = ["JNLC", "CSW", "PTC"]

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce, QueryOrderStatus
    from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


class AlpacaBroker(BrokerBase, BrokerHistory):
    """Broker implementation backed by the Alpaca Trading API.

    Supports paper trading (paper=True, default) and live trading (paper=False).
    Always start with paper=True and validate the full execution loop before
    switching to live.

    Implements both BrokerBase (operational) and BrokerHistory (analytics).
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required. Install it with: pip install alpaca-py"
            )
        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper

    # ------------------------------------------------------------------
    # BrokerBase — operational
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit a market order and return the Alpaca order ID."""
        side = OrderSide.BUY if order.operation_type == OPERATION_BUY else OrderSide.SELL
        request = MarketOrderRequest(
            symbol=order.ticker,
            qty=order.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        alpaca_order = self._client.submit_order(request)
        return str(alpaca_order.id)

    def get_fill(self, order_id: str) -> Optional[Fill]:
        """Return a Fill once the order is fully filled, None if still pending."""
        alpaca_order = self._client.get_order_by_id(order_id)
        if alpaca_order.status != OrderStatus.FILLED:
            return None
        side = (
            OPERATION_BUY
            if alpaca_order.side == OrderSide.BUY
            else OPERATION_SELL
        )
        return Fill(
            ticker=alpaca_order.symbol,
            qty=float(alpaca_order.filled_qty),
            fill_price=float(alpaca_order.filled_avg_price),
            operation_type=side,
            fill_time=alpaca_order.filled_at,
            broker_order_id=order_id,
        )

    def get_positions(self) -> Portfolio:
        """Return current Alpaca positions as a Portfolio."""
        positions = self._client.get_all_positions()
        portfolio = Portfolio.create_empty()
        for pos in positions:
            qty = float(pos.qty)
            if qty > 0:
                portfolio.modify_position(pos.symbol, qty)
        return portfolio

    def get_cash(self) -> float:
        """Return current cash balance from the Alpaca account."""
        return float(self._client.get_account().cash)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if the cancellation succeeded."""
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def is_market_open(self) -> bool:
        """Return True if the NYSE is currently open for trading."""
        clock = self._client.get_clock()
        return clock.is_open

    # ------------------------------------------------------------------
    # BrokerHistory — analytics
    # ------------------------------------------------------------------

    @property
    def supports_cash_flow_history(self) -> bool:
        return True

    def get_order_history(self, start: dt.datetime, end: dt.datetime) -> List[Fill]:
        """Return all filled orders between start and end."""
        try:
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=start,
                until=end,
                limit=500,
            )
            orders = self._client.get_orders(filter=request)
        except Exception as exc:
            log.error("Failed to fetch order history from Alpaca: %s", exc)
            raise

        fills = []
        for o in orders:
            if o.status != OrderStatus.FILLED:
                continue
            side = OPERATION_BUY if o.side == OrderSide.BUY else OPERATION_SELL
            fills.append(Fill(
                ticker=o.symbol,
                qty=float(o.filled_qty),
                fill_price=float(o.filled_avg_price),
                operation_type=side,
                fill_time=o.filled_at,
                broker_order_id=str(o.id),
            ))
        return fills

    def get_cash_flows(self, start: dt.datetime, end: dt.datetime) -> List[CashFlow]:
        """Return deposits and withdrawals between start and end."""
        try:
            activities = self._client.get_account_activities(
                activity_types=_CASH_FLOW_ACTIVITY_TYPES,
            )
        except Exception as exc:
            log.error("Failed to fetch cash flow history from Alpaca: %s", exc)
            raise

        start_ts = _to_utc(start)
        end_ts = _to_utc(end)

        cash_flows = []
        for a in activities:
            raw_date = getattr(a, "date", None) or getattr(a, "transaction_time", None)
            if raw_date is None:
                continue
            event_date = _to_utc(raw_date)
            if not (start_ts <= event_date <= end_ts):
                continue
            amount = float(getattr(a, "net_amount", 0.0) or 0.0)
            cash_flows.append(CashFlow(
                date=event_date,
                amount=amount,
                description=getattr(a, "description", None) or str(a.activity_type),
            ))
        return cash_flows


def _to_utc(value) -> dt.datetime:
    """Normalise a date/datetime/Timestamp to a tz-aware UTC datetime."""
    import pandas as pd
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        value = dt.datetime(value.year, value.month, value.day)
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value
