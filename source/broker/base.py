import time
from abc import ABC, abstractmethod
from typing import Optional

from resources import Fill, Order, Portfolio


class BrokerBase(ABC):
    """Abstract interface for broker integrations.

    Concrete implementations: AlpacaBroker (live/paper API) and
    PaperBroker (in-memory, for testing and backtesting).
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
