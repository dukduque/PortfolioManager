"""Abstract base class for all market data providers.

Every concrete implementation must satisfy the three-method contract used
throughout the codebase (backtest.py, account_manager.py, run_daily.py).
FakeDataManager in tests/ also subclasses this so type-checking catches drift.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def quotient_diff(series: pd.Series) -> pd.Series:
    """Compute gross return ratios: price[t+1] / price[t].

    A value of 1.05 means +5%; 0.95 means -5%.  This is the return
    representation used throughout the CVaR optimiser.
    """
    y = np.array(series)
    return pd.Series(y[1:] / y[:-1], index=series.index[1:])


EMPTY_METADATA = {
    "name": "",
    "sector": "",
    "subsector": "",
    "market_cap": 0,
}


class DataManagerBase(ABC):
    """Contract that every data provider must implement.

    All three methods are used by backtest.py, account_manager.py, and
    run_daily.py.  The FakeDataManager in tests/ satisfies the same interface
    so unit tests remain hermetic.

    get_prices  — full price history for the requested tickers
    get_returns — gross-return ratios for a date window, outliers removed
    get_metadata — sector / name info (best-effort; may return EMPTY_METADATA)
    """

    @abstractmethod
    def get_prices(self, assets=None) -> pd.DataFrame:
        """Return a DataFrame of adjusted close prices.

        Args:
            assets: ticker string, list of tickers, or None for all available.

        Returns:
            DataFrame with DatetimeIndex (rows = trading days) and ticker
            columns.  Callers are responsible for slicing to their date range.
        """

    @abstractmethod
    def get_returns(
        self,
        start_date,
        end_date,
        stocks=None,
        outlier_return: float = 10,
    ) -> pd.DataFrame:
        """Return gross-return ratios for *stocks* over [start_date, end_date].

        Args:
            start_date: inclusive start (datetime or date).
            end_date:   inclusive end   (datetime or date).
            stocks:     list of tickers to include; None / [] = all available.
            outlier_return: rows where any return exceeds this ratio are
                            dropped (default 10 = 900 % gain in one day).

        Returns:
            DataFrame with DatetimeIndex and ticker columns.  Any ticker that
            has NaN in the window is dropped entirely.
        """

    @abstractmethod
    def get_metadata(self, asset: str) -> dict:
        """Return a metadata dict for *asset*.

        Keys: name, sector, subsector, market_cap.
        Implementations that cannot provide a field return EMPTY_METADATA
        values (empty string / 0) rather than raising.
        """
