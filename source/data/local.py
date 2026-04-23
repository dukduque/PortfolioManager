"""LocalDataManager — pickle-file backed data provider.

This is the original DataManager from database_handler.py, promoted to a
first-class implementation of DataManagerBase.  It exists to support the
manual / interactive workflow (Jupyter notebooks, ad-hoc rebalancing) where
the price database is a local close.pkl file kept up-to-date by
database_handler.update_database().

For the automated daily job use AlpacaDataManager instead.
"""
import logging

import pandas as pd

from data.base import DataManagerBase, EMPTY_METADATA, quotient_diff

log = logging.getLogger(__name__)


class LocalDataManager(DataManagerBase):
    """Read prices and metadata from local pickle files.

    Args:
        db_file:       path (str or Path) to the close-price pickle, e.g.
                       "data/close.pkl".
        metadata_file: path to the metadata pickle, e.g. "data/metadata.pkl".

    The pickle files are loaded once at construction time.  Call
    database_handler.update_database() separately to refresh them.
    """

    def __init__(self, db_file="close.pkl", metadata_file="metadata.pkl"):
        # Import here to avoid a circular dependency: database_handler imports
        # LocalDataManager, so we defer the heavy imports to call time.
        from database_handler import load_database, get_tickers_metadata
        self.db = load_database(db_file)
        self.metadata = get_tickers_metadata(metadata_file)
        self.db_file = db_file
        self.metadata_file = metadata_file
        self._returns_cache: dict = {}

    def get_prices(self, assets=None) -> pd.DataFrame:
        if assets is None or (hasattr(assets, '__len__') and len(assets) == 0):
            return self.db

        if isinstance(assets, str):
            assets = [assets]

        missing = [a for a in assets if a not in self.db.columns]
        if missing:
            from database_handler import update_database_single_stock
            for asset in missing:
                try:
                    self.db = update_database_single_stock(
                        self.db, asset, self.db_file, self.metadata_file
                    )
                except Exception:
                    log.warning("Could not fetch prices for %s", asset)

        out = pd.DataFrame(index=self.db.index, columns=assets)
        out.update(self.db)
        return out

    def get_returns(
        self,
        start_date,
        end_date,
        stocks=None,
        outlier_return: float = 10,
    ) -> pd.DataFrame:
        stocks = stocks or []
        cache_key = (start_date, end_date)
        if not stocks and cache_key in self._returns_cache:
            return self._returns_cache[cache_key]

        for s in stocks:
            if s not in self.db.columns:
                try:
                    self.get_prices(s)
                except Exception:
                    log.warning("Failed to obtain data for %s", s)

        db = self.db[
            (self.db.index >= pd.Timestamp(start_date)) &
            (self.db.index <= pd.Timestamp(end_date))
        ]
        db = db.dropna(axis=0, how="all").dropna(axis=1)
        if len(db) < 2:
            return pd.DataFrame()

        db_r = db.apply(quotient_diff, axis=0)
        db_r = db_r[db_r < outlier_return].dropna(axis=0)

        if not stocks:
            self._returns_cache[cache_key] = db_r
        return db_r

    def get_metadata(self, asset: str) -> dict:
        import yfinance as yf
        from database_handler import save_metadata
        assert isinstance(asset, str)
        if asset in self.metadata:
            return self.metadata[asset]
        try:
            info = yf.Ticker(asset).info
            meta = EMPTY_METADATA.copy()
            meta["name"] = info.get("shortName", asset)
            meta["sector"] = info.get("sector", "")
            meta["subsector"] = info.get("industry", "")
            meta["market_cap"] = info.get("marketCap", 0)
            self.metadata[asset] = meta
            save_metadata(self.metadata, self.metadata_file)
        except Exception:
            self.metadata[asset] = {**EMPTY_METADATA, "name": asset}
        return self.metadata[asset]
