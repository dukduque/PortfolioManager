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
from typing import Optional

from broker.base import BrokerBase
from resources import Fill, Order, Portfolio, OPERATION_BUY, OPERATION_SELL

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


class AlpacaBroker(BrokerBase):
    """Broker implementation backed by the Alpaca Trading API.

    Supports paper trading (paper=True, default) and live trading (paper=False).
    Always start with paper=True and validate the full execution loop before
    switching to live.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        if not _ALPACA_AVAILABLE:
            raise ImportError(
                "alpaca-py is required. Install it with: pip install alpaca-py"
            )
        self._client = TradingClient(api_key, secret_key, paper=paper)
        self._paper = paper

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
