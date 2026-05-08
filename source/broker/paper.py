import datetime as dt
from typing import Dict, List, Optional

from broker.base import BrokerBase, BrokerHistory
from resources import CashFlow, Fill, Order, Portfolio, OPERATION_BUY, OPERATION_SELL


class PaperBroker(BrokerBase, BrokerHistory):
    """In-memory broker that fills orders instantly at the order price.

    No network calls, no credentials. Used for hermetic testing and
    for driving the walk-forward backtester.

    Implements both BrokerBase (operational) and BrokerHistory (analytics)
    from in-memory state, so backtests can reconstruct an Account without
    a real broker connection.
    """

    def __init__(self, initial_portfolio: Optional[Portfolio] = None, initial_cash: float = 0.0):
        self._portfolio = initial_portfolio or Portfolio.create_empty()
        self._cash = initial_cash
        self._fills: Dict[str, Fill] = {}
        self._fill_history: List[Fill] = []
        self._cash_flows: List[CashFlow] = []
        self._next_id = 0
        self._market_open = True

    # ------------------------------------------------------------------
    # BrokerBase — operational
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Fill the order immediately at order.price and update positions."""
        if order.operation_type not in (OPERATION_BUY, OPERATION_SELL):
            raise ValueError(f"Unsupported operation type: {order.operation_type}")

        new_portfolio = Portfolio.create_from_transaction(self._portfolio, [order])
        if new_portfolio is None:
            raise ValueError(
                f"Invalid order {order}: sell quantity may exceed current holdings."
            )
        self._portfolio = new_portfolio

        if order.operation_type == OPERATION_BUY:
            self._cash -= order.qty * order.price
        else:
            self._cash += order.qty * order.price

        self._next_id += 1
        order_id = f"paper-{self._next_id}"
        fill = Fill(
            ticker=order.ticker,
            qty=order.qty,
            fill_price=order.price,
            operation_type=order.operation_type,
            fill_time=dt.datetime.now(dt.timezone.utc),
            broker_order_id=order_id,
        )
        self._fills[order_id] = fill
        self._fill_history.append(fill)
        return order_id

    def get_fill(self, order_id: str) -> Optional[Fill]:
        return self._fills.get(order_id)

    def get_positions(self) -> Portfolio:
        return self._portfolio

    def get_cash(self) -> float:
        return self._cash

    def cancel_order(self, order_id: str) -> bool:
        # PaperBroker fills are immediate; there is nothing left to cancel.
        return False

    def is_market_open(self) -> bool:
        return self._market_open

    def set_market_open(self, is_open: bool) -> None:
        """Test helper: simulate market open/closed state."""
        self._market_open = is_open

    def deposit(self, amount: float, date: Optional[dt.datetime] = None) -> None:
        """Record a cash deposit. Used by the backtester to simulate funding."""
        self._cash += amount
        self._cash_flows.append(CashFlow(
            date=date or dt.datetime.now(dt.timezone.utc),
            amount=amount,
            description="deposit",
        ))

    def withdraw(self, amount: float, date: Optional[dt.datetime] = None) -> None:
        """Record a cash withdrawal."""
        self._cash -= amount
        self._cash_flows.append(CashFlow(
            date=date or dt.datetime.now(dt.timezone.utc),
            amount=-amount,
            description="withdrawal",
        ))

    # ------------------------------------------------------------------
    # BrokerHistory — analytics
    # ------------------------------------------------------------------

    @property
    def supports_cash_flow_history(self) -> bool:
        return True

    def get_order_history(
        self, start: dt.datetime, end: dt.datetime
    ) -> List[Fill]:
        """Return all fills recorded between start and end."""
        return [
            f for f in self._fill_history
            if start <= f.fill_time <= end
        ]

    def get_cash_flows(
        self, start: dt.datetime, end: dt.datetime
    ) -> List[CashFlow]:
        """Return all cash flows recorded between start and end."""
        return [
            cf for cf in self._cash_flows
            if start <= cf.date <= end
        ]


class CostAwarePaperBroker(PaperBroker):
    """PaperBroker that applies a bid-ask spread haircut to each fill price.

    Buys are filled at  price * (1 + cost_bps / 10_000).
    Sells are filled at price * (1 - cost_bps / 10_000).
    The position qty is unchanged; the cost is captured in the fill price
    and therefore in the cash accounting.
    """

    def __init__(
        self,
        cost_bps: float = 5.0,
        initial_portfolio: Optional[Portfolio] = None,
        initial_cash: float = 0.0,
    ):
        super().__init__(initial_portfolio=initial_portfolio, initial_cash=initial_cash)
        self._cost_bps = cost_bps

    def submit_order(self, order: Order) -> str:
        cost_factor = self._cost_bps / 10_000
        if order.operation_type == OPERATION_BUY:
            adjusted_price = order.price * (1 + cost_factor)
        elif order.operation_type == OPERATION_SELL:
            adjusted_price = order.price * (1 - cost_factor)
        else:
            adjusted_price = order.price
        adjusted = Order(order.ticker, order.qty, adjusted_price, order.operation_type)
        return super().submit_order(adjusted)
