"""AlpacaDataManager — market data via Alpaca + local parquet cache.

Architecture
------------
Historical price data is stored in a per-ticker parquet cache:

    cache_dir/
        AAPL.parquet   # DatetimeIndex, single column "close" (adjusted)
        MSFT.parquet
        SPY.parquet
        ...

refresh_cache(tickers, end_date)
    Called once per day (before the optimiser runs).  Finds the last cached
    date for each ticker, fetches only the missing tail from Alpaca in a
    single bulk API call, and appends to each file.  For a warm cache this
    fetches one row per ticker — equivalent latency to the old pickle update.

get_prices / get_returns
    Read entirely from the parquet cache.  No network call on the hot path.

get_metadata
    Returns name + exchange from Alpaca's asset endpoint.  Sector and
    market_cap are not available from Alpaca; those fields return "Unknown".

Dependency injection
--------------------
Pass _data_client= and _trading_client= in the constructor to supply mock
clients for testing without real API credentials.
"""
import datetime as dt
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data.base import DataManagerBase, EMPTY_METADATA, quotient_diff

log = logging.getLogger(__name__)

# Default start date for a cold-cache fetch (no prior history on disk).
_DEFAULT_HISTORY_START = dt.date(2015, 1, 1)


def _to_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    return value


class AlpacaDataManager(DataManagerBase):
    """Alpaca Market Data API backed by a local parquet cache.

    Args:
        api_key:        Alpaca API key (same key used for trading).
        secret_key:     Alpaca secret key.
        cache_dir:      Directory for per-ticker parquet files.
                        Created automatically if absent.
        _data_client:   Override the StockHistoricalDataClient (testing).
        _trading_client:Override the TradingClient used for get_metadata
                        (testing).  Pass None to skip metadata lookups.
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        cache_dir: Path = Path("data/cache/alpaca"),
        _data_client=None,
        _trading_client=None,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._meta_cache: dict = {}

        if _data_client is not None:
            self._data_client = _data_client
        else:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._data_client = StockHistoricalDataClient(api_key, secret_key)
            except ImportError:
                raise ImportError(
                    "alpaca-py is required for AlpacaDataManager. "
                    "Install it with: pip install alpaca-py"
                )

        # Trading client for get_metadata; optional — skip if not provided.
        if _trading_client is not None:
            self._trading_client = _trading_client
        elif api_key:
            try:
                from alpaca.trading.client import TradingClient
                self._trading_client = TradingClient(api_key, secret_key, paper=True)
            except Exception:
                self._trading_client = None
        else:
            self._trading_client = None

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        return self._cache_dir / f"{ticker}.parquet"

    def _read_cache(self, ticker: str) -> Optional[pd.DataFrame]:
        """Return cached DataFrame for *ticker*, or None if not cached."""
        path = self._cache_path(ticker)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def _build_bars_request(self, tickers, start, end):
        """Build a StockBarsRequest, falling back to SimpleNamespace for testing."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            from alpaca.data.enums import Adjustment
            return StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment=Adjustment.ALL,
            )
        except ImportError:
            from types import SimpleNamespace
            return SimpleNamespace(
                symbol_or_symbols=tickers, start=start, end=end
            )

    def _write_cache(self, ticker: str, df: pd.DataFrame) -> None:
        """Atomically write *df* to the ticker's parquet file."""
        path = self._cache_path(ticker)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.replace(path)   # atomic on same filesystem

    def refresh_cache(
        self,
        tickers: List[str],
        end_date=None,
        force: bool = False,
    ) -> None:
        """Fetch missing price history for *tickers* and update the cache.

        This is called by run_daily.py once per trading day, before the CVaR
        optimiser runs.  It finds the last cached date for each ticker and
        fetches only the missing tail from Alpaca in a single bulk API call.
        For a warm cache this is typically one row per ticker.

        Corporate actions (splits, spin-offs) invalidate historical adjusted-
        close prices retroactively.  When Alpaca re-issues corrected bars the
        cached series becomes inconsistent at the join point.  Pass
        ``force=True`` for any affected tickers to delete their cache files
        and re-download the full history from _DEFAULT_HISTORY_START.

        Args:
            tickers:  List of ticker symbols to refresh.
            end_date: Last date to fetch (inclusive).  Defaults to yesterday
                      so we never request an incomplete trading day.
            force:    If True, delete the existing cache file for each ticker
                      before fetching, forcing a full re-download.  Use after
                      a stock split or other corporate action.
        """
        if end_date is None:
            end_date = dt.date.today() - dt.timedelta(days=1)
        end_date = _to_date(end_date)

        if force:
            for ticker in tickers:
                path = self._cache_path(ticker)
                if path.exists():
                    path.unlink()
                    log.info("Deleted cache for %s (force=True).", ticker)

        # Determine the fetch window for each ticker.
        fetch_start: dict[str, dt.date] = {}
        for ticker in tickers:
            cached = self._read_cache(ticker)
            if cached is None or cached.empty:
                fetch_start[ticker] = _DEFAULT_HISTORY_START
            else:
                last = _to_date(cached.index[-1])
                if last < end_date:
                    # Start one calendar day after the last cached date.
                    fetch_start[ticker] = last + dt.timedelta(days=1)
                # else: already up-to-date; skip

        if not fetch_start:
            log.debug("Cache already up-to-date for all %d tickers.", len(tickers))
            return

        # One bulk call from the earliest needed start date.
        bulk_start = min(fetch_start.values())
        log.info(
            "Fetching %d ticker(s) from Alpaca: %s → %s",
            len(fetch_start), bulk_start, end_date,
        )

        try:
            request = self._build_bars_request(
                list(fetch_start.keys()),
                dt.datetime.combine(bulk_start, dt.time.min),
                dt.datetime.combine(end_date, dt.time.max),
            )
            bars = self._data_client.get_stock_bars(request)
            raw = bars.df  # MultiIndex (symbol, timestamp)
        except Exception as exc:
            log.error("Alpaca data fetch failed: %s", exc)
            raise

        if raw.empty:
            log.warning("Alpaca returned no data for the requested range.")
            return

        # Append new rows to each ticker's parquet file.
        symbols_returned = raw.index.get_level_values("symbol").unique()
        for ticker in fetch_start:
            if ticker not in symbols_returned:
                log.warning("Alpaca returned no data for %s.", ticker)
                continue

            new_rows = (
                raw.loc[ticker][["close"]]
                .rename_axis("timestamp")
            )
            new_rows.index = pd.DatetimeIndex(new_rows.index).normalize()

            # Drop rows earlier than the ticker's personal fetch start.
            ticker_start = pd.Timestamp(fetch_start[ticker])
            new_rows = new_rows[new_rows.index >= ticker_start]

            cached = self._read_cache(ticker)
            if cached is not None and not cached.empty:
                new_rows = new_rows[new_rows.index > cached.index[-1]]
                combined = pd.concat([cached, new_rows]).sort_index()
            else:
                combined = new_rows.sort_index()

            combined = combined[~combined.index.duplicated(keep="last")]
            self._write_cache(ticker, combined)
            log.debug("Cached %d new rows for %s.", len(new_rows), ticker)

    # ------------------------------------------------------------------
    # DataManagerBase implementation
    # ------------------------------------------------------------------

    def get_prices(self, assets=None) -> pd.DataFrame:
        """Read adjusted close prices from the parquet cache.

        Does NOT fetch from Alpaca — call refresh_cache() first to ensure
        the cache is current.

        Args:
            assets: ticker string, list of tickers, or None for all cached.

        Returns:
            DataFrame with DatetimeIndex and ticker columns.
        """
        if assets is None:
            assets = [p.stem for p in self._cache_dir.glob("*.parquet")]
            if not assets:
                raise RuntimeError(
                    f"No parquet cache files found in '{self._cache_dir}'. "
                    "Call refresh_cache() before get_prices()."
                )

        if isinstance(assets, str):
            assets = [assets]

        frames = {}
        for ticker in assets:
            cached = self._read_cache(ticker)
            if cached is None or cached.empty:
                log.warning(
                    "No cached data for %s. Call refresh_cache() first.", ticker
                )
                continue
            frames[ticker] = cached["close"]

        if not frames:
            return pd.DataFrame()

        return pd.DataFrame(frames).sort_index()

    def get_metadata(self, asset: str) -> dict:
        """Return asset name and exchange from Alpaca; sector is unavailable.

        Alpaca's market data API does not provide sector or market cap.
        Those fields are returned as empty strings / 0.  Use LocalDataManager
        or supplement with another provider if fundamentals are required.
        """
        if asset in self._meta_cache:
            return self._meta_cache[asset]

        meta = {**EMPTY_METADATA, "name": asset}

        if self._trading_client is not None:
            try:
                asset_info = self._trading_client.get_asset(asset)
                meta["name"] = asset_info.name or asset
                meta["sector"] = getattr(asset_info, "exchange", "") or ""
            except Exception as exc:
                log.debug("Could not fetch metadata for %s from Alpaca: %s", asset, exc)

        self._meta_cache[asset] = meta
        return meta
