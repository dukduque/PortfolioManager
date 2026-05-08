import time
from abc import ABC, abstractmethod
from typing import List, Optional

from resources import CashFlow, Fill, Order, Portfolio


class BrokerBase(ABC):
    """Operational broker interface — required for rebalancing.

    Every broker integration must implement all abstract methods here.
    This is the only interface run_daily.py depends on.
    """

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submit an order and return the broker-assigned order ID."""

    @abstractmethod
    def get_fill(self, order_id: str) -> Optional[Fill]:
        """Return a Fill if the order is fully filled, None if still pending."""

    @abstractmethod
    def get_positions(self) -> Portfolio:
        """Return current holdings as a Portfolio."""

    @abstractmethod
    def get_cash(self) -> float:
        """Return current cash balance available for trading."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True if the cancellation succeeded."""

    @abstractmethod
    def is_market_open(self) -> bool:
        """Return True if the market is currently open for trading."""

    def submit_and_await_fill(
        self,
        order: Order,
        timeout_seconds: int = 60,
        poll_interval: int = 2,
    ) -> Optional[Fill]:
        """Submit an order and poll until filled or timeout expires.

        Returns the Fill on success, None if timeout is reached.
        """
        order_id = self.submit_order(order)
        elapsed = 0
        while elapsed < timeout_seconds:
            fill = self.get_fill(order_id)
            if fill is not None:
                return fill
            time.sleep(poll_interval)
            elapsed += poll_interval
        return None


class BrokerHistory(ABC):
    """Analytics broker interface — required for equity curve and benchmarking.

    Brokers that support historical data implement this alongside BrokerBase.
    Callers check isinstance(broker, BrokerHistory) before using these methods.

    get_order_history is always required for analytics. get_cash_flows is
    optional — brokers opt in by overriding supports_cash_flow_history and
    get_cash_flows. Without cash flow history the benchmark curve is unavailable.
    """

    @abstractmethod
    def get_order_history(
        self, start: "datetime", end: "datetime"
    ) -> List[Fill]:
        """Return all filled orders between start and end as Fill objects."""

    @property
    def supports_cash_flow_history(self) -> bool:
        """True if this broker can return deposit/withdrawal history."""
        return False

    def get_cash_flows(
        self, start: "datetime", end: "datetime"
    ) -> List[CashFlow]:
        """Return deposits and withdrawals between start and end.

        Raises NotImplementedError for brokers that do not support this.
        Override supports_cash_flow_history to return True when implementing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support cash flow history. "
            "Check supports_cash_flow_history before calling get_cash_flows()."
        )
