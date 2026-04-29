from tests import import_source_modules
import_source_modules()

import datetime as dt
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from data.alpaca import AlpacaDataManager
from data.base import DataManagerBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar_df(tickers, dates) -> pd.DataFrame:
    """Build a fake Alpaca bars MultiIndex DataFrame."""
    idx = pd.MultiIndex.from_product(
        [tickers, pd.DatetimeIndex(dates)],
        names=["symbol", "timestamp"],
    )
    rng = np.random.default_rng(0)
    prices = 100.0 + rng.normal(0, 2, len(idx)).cumsum()
    prices = np.abs(prices) + 10   # ensure positive
    return pd.DataFrame({"close": prices}, index=idx)


def _fake_bars(tickers, dates):
    """Return a mock object whose .df is the bar DataFrame."""
    mock = MagicMock()
    mock.df = _make_bar_df(tickers, dates)
    return mock


def _make_dm(tmp_path, tickers, dates, trading_client=None):
    """Build an AlpacaDataManager backed by a mock data client."""
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = _fake_bars(tickers, dates)
    return AlpacaDataManager(
        cache_dir=tmp_path,
        _data_client=mock_client,
        _trading_client=trading_client,
    )


TICKERS = ["AA", "BB", "CC"]
DATES = pd.bdate_range("2023-01-02", periods=20)


# ---------------------------------------------------------------------------
# DataManagerBase contract
# ---------------------------------------------------------------------------

def test_alpaca_data_manager_is_data_manager_base():
    assert issubclass(AlpacaDataManager, DataManagerBase)


# ---------------------------------------------------------------------------
# refresh_cache
# ---------------------------------------------------------------------------

def test_refresh_cache_creates_parquet_files(tmp_path):
    dm = _make_dm(tmp_path, TICKERS, DATES)
    dm.refresh_cache(TICKERS, end_date=DATES[-1].date())

    for ticker in TICKERS:
        assert (tmp_path / f"{ticker}.parquet").exists()


def test_refresh_cache_parquet_has_close_column(tmp_path):
    dm = _make_dm(tmp_path, TICKERS, DATES)
    dm.refresh_cache(TICKERS, end_date=DATES[-1].date())

    cached = pd.read_parquet(tmp_path / "AA.parquet")
    assert "close" in cached.columns


def test_refresh_cache_does_not_refetch_up_to_date_tickers(tmp_path):
    dm = _make_dm(tmp_path, TICKERS, DATES)
    dm.refresh_cache(TICKERS, end_date=DATES[-1].date())

    call_count_after_first = dm._data_client.get_stock_bars.call_count

    # Second refresh with same end_date — nothing is stale.
    dm.refresh_cache(TICKERS, end_date=DATES[-1].date())
    assert dm._data_client.get_stock_bars.call_count == call_count_after_first


def test_refresh_cache_incremental_appends_only_new_rows(tmp_path):
    dates_first = DATES[:10]
    dates_second = DATES[10:]

    dm = _make_dm(tmp_path, ["AA"], dates_first)
    dm.refresh_cache(["AA"], end_date=dates_first[-1].date())

    first_len = len(pd.read_parquet(tmp_path / "AA.parquet"))

    # Simulate a second day's worth of data.
    dm._data_client.get_stock_bars.return_value = _fake_bars(["AA"], dates_second)
    dm.refresh_cache(["AA"], end_date=dates_second[-1].date())

    total_len = len(pd.read_parquet(tmp_path / "AA.parquet"))
    assert total_len == first_len + len(dates_second)


def test_refresh_cache_atomic_write_leaves_no_tmp_files(tmp_path):
    dm = _make_dm(tmp_path, ["AA"], DATES)
    dm.refresh_cache(["AA"], end_date=DATES[-1].date())

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []


def test_refresh_cache_force_deletes_and_refetches(tmp_path):
    """force=True must delete the existing cache and re-download from scratch."""
    dm = _make_dm(tmp_path, ["AA"], DATES[:10])
    dm.refresh_cache(["AA"], end_date=DATES[9].date())

    first_call_count = dm._data_client.get_stock_bars.call_count

    # Without force, a second refresh for the same end_date is a no-op.
    dm.refresh_cache(["AA"], end_date=DATES[9].date())
    assert dm._data_client.get_stock_bars.call_count == first_call_count

    # With force=True the cache file is deleted and re-fetched.
    dm.refresh_cache(["AA"], end_date=DATES[9].date(), force=True)
    assert dm._data_client.get_stock_bars.call_count == first_call_count + 1


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------

def test_get_prices_raises_when_cache_empty_and_assets_none(tmp_path):
    """get_prices(None) must raise if no parquet files exist yet."""
    dm = _make_dm(tmp_path, ["AA"], DATES)  # no refresh_cache called
    with pytest.raises(RuntimeError, match="refresh_cache"):
        dm.get_prices()


def test_get_prices_returns_dataframe_after_refresh(tmp_path):
    dm = _make_dm(tmp_path, TICKERS, DATES)
    dm.refresh_cache(TICKERS, end_date=DATES[-1].date())

    prices = dm.get_prices(TICKERS)
    assert isinstance(prices, pd.DataFrame)
    assert set(prices.columns) == set(TICKERS)


def test_get_prices_date_index_is_datetime(tmp_path):
    dm = _make_dm(tmp_path, ["AA"], DATES)
    dm.refresh_cache(["AA"], end_date=DATES[-1].date())

    prices = dm.get_prices(["AA"])
    assert isinstance(prices.index, pd.DatetimeIndex)


def test_get_prices_single_ticker_string(tmp_path):
    dm = _make_dm(tmp_path, ["AA"], DATES)
    dm.refresh_cache(["AA"], end_date=DATES[-1].date())

    prices = dm.get_prices("AA")
    assert "AA" in prices.columns


def test_get_prices_empty_when_not_cached(tmp_path):
    dm = _make_dm(tmp_path, ["AA"], DATES)
    prices = dm.get_prices(["NOTCACHED"])
    assert prices.empty


# ---------------------------------------------------------------------------
# get_returns
# ---------------------------------------------------------------------------

def test_get_returns_shape(tmp_path):
    dm = _make_dm(tmp_path, TICKERS, DATES)
    dm.refresh_cache(TICKERS, end_date=DATES[-1].date())

    returns = dm.get_returns(DATES[0], DATES[-1], stocks=TICKERS)
    # Returns has n-1 rows compared to prices in the window.
    assert len(returns) > 0
    assert set(returns.columns) == set(TICKERS)


def test_get_returns_values_are_positive(tmp_path):
    """Gross return ratios must be > 0 for any positive price series."""
    dm = _make_dm(tmp_path, ["AA"], DATES)
    dm.refresh_cache(["AA"], end_date=DATES[-1].date())

    returns = dm.get_returns(DATES[0], DATES[-1])
    assert (returns > 0).all().all()


def test_get_returns_empty_for_short_window(tmp_path):
    dm = _make_dm(tmp_path, ["AA"], DATES)
    dm.refresh_cache(["AA"], end_date=DATES[-1].date())

    # One-day window → fewer than 2 prices → empty returns.
    returns = dm.get_returns(DATES[0], DATES[0])
    assert returns.empty


def test_get_returns_outlier_filter_applied(tmp_path):
    """Returns exceeding outlier_return threshold must be dropped."""
    dm = _make_dm(tmp_path, ["AA"], DATES)
    dm.refresh_cache(["AA"], end_date=DATES[-1].date())

    # With threshold = 0.01, almost every return exceeds it → nearly empty.
    returns = dm.get_returns(DATES[0], DATES[-1], outlier_return=0.01)
    assert returns.empty or len(returns) < len(DATES) - 1


# ---------------------------------------------------------------------------
# get_metadata
# ---------------------------------------------------------------------------

def test_get_metadata_returns_dict_with_required_keys(tmp_path):
    dm = AlpacaDataManager(cache_dir=tmp_path, _data_client=MagicMock(),
                           _trading_client=None)
    meta = dm.get_metadata("AAPL")
    for key in ("name", "sector", "subsector", "market_cap"):
        assert key in meta


def test_get_metadata_uses_trading_client_name(tmp_path):
    mock_trading = MagicMock()
    mock_asset = MagicMock()
    mock_asset.name = "Apple Inc."
    mock_asset.exchange = "NASDAQ"
    mock_trading.get_asset.return_value = mock_asset

    dm = AlpacaDataManager(cache_dir=tmp_path, _data_client=MagicMock(),
                           _trading_client=mock_trading)
    meta = dm.get_metadata("AAPL")
    assert meta["name"] == "Apple Inc."


def test_get_metadata_cached_after_first_call(tmp_path):
    mock_trading = MagicMock()
    mock_trading.get_asset.return_value = MagicMock(name="X", exchange="NYSE")

    dm = AlpacaDataManager(cache_dir=tmp_path, _data_client=MagicMock(),
                           _trading_client=mock_trading)
    dm.get_metadata("AAPL")
    dm.get_metadata("AAPL")
    assert mock_trading.get_asset.call_count == 1


def test_get_metadata_graceful_on_trading_client_failure(tmp_path):
    mock_trading = MagicMock()
    mock_trading.get_asset.side_effect = RuntimeError("API down")

    dm = AlpacaDataManager(cache_dir=tmp_path, _data_client=MagicMock(),
                           _trading_client=mock_trading)
    meta = dm.get_metadata("AAPL")   # must not raise
    assert meta["name"] == "AAPL"    # falls back to ticker as name
