"""Fake data helpers for hermetic testing.

FakeDataManager accepts a pre-built price DataFrame and implements the same
interface as DataManagerBase without any disk I/O or network calls.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "source"))

import numpy as np
import pandas as pd

from data.base import DataManagerBase, EMPTY_METADATA, quotient_diff


class FakeDataManager(DataManagerBase):
    """In-memory DataManagerBase backed by a synthetic price DataFrame.

    Args:
        prices: DataFrame with tickers as columns and dates as index.
                All prices must be positive floats.
    """

    def __init__(self, prices: pd.DataFrame):
        self._prices = prices.copy()

    def get_prices(self, assets=None) -> pd.DataFrame:
        if assets is None:
            return self._prices
        if isinstance(assets, str):
            assets = [assets]
        cols = [a for a in assets if a in self._prices.columns]
        return self._prices[cols]

    def get_metadata(self, asset: str) -> dict:
        return {**EMPTY_METADATA, "name": asset, "sector": "Unknown"}


def make_prices(
    tickers,
    n_days: int = 60,
    start: str = "2023-01-02",
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic price DataFrame for testing.

    Uses a seeded random walk so results are reproducible.  The first ticker
    gets a base price of 100, each subsequent one adds 20.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    prices = {}
    for i, ticker in enumerate(tickers):
        base = 100.0 + i * 20.0
        daily_ret = rng.normal(0.0005, 0.015, n_days)
        prices[ticker] = base * np.cumprod(1 + daily_ret)
    return pd.DataFrame(prices, index=dates)
