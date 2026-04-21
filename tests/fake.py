"""Fake data helpers for hermetic testing.

FakeDataManager accepts a pre-built price DataFrame and implements the same
interface as DataManager (get_prices, get_returns, get_metadata) without any
disk I/O or network calls.  Import it in test modules that need price data.
"""
import numpy as np
import pandas as pd


def _quotient_diff(series):
    """Same return computation as database_handler.quotien_diff."""
    y = np.array(series)
    return pd.Series(y[1:] / y[:-1], index=series.index[1:])


class FakeDataManager:
    """In-memory DataManager backed by a synthetic price DataFrame.

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

    def get_returns(
        self, start_date, end_date, stocks=None, outlier_return=10
    ) -> pd.DataFrame:
        df = self._prices[
            (self._prices.index >= pd.Timestamp(start_date))
            & (self._prices.index <= pd.Timestamp(end_date))
        ]
        if stocks:
            available = [s for s in stocks if s in df.columns]
            df = df[available] if available else df
        df = df.dropna(axis=1)
        if len(df) < 2:
            return pd.DataFrame()
        db_r = df.apply(_quotient_diff, axis=0)
        db_r = db_r[db_r < outlier_return].dropna(axis=0)
        return db_r

    def get_metadata(self, asset: str) -> dict:
        return {
            "name": asset,
            "sector": "Unknown",
            "subsector": "",
            "market_cap": 0,
        }


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
