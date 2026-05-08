"""Daily rebalancing entry point.

Run this script once a day (e.g. via cron or systemd timer at 10:00 ET).
It will:
  1. Exit cleanly if today is not a NYSE trading day.
  2. Exit cleanly if the job already succeeded today (idempotency).
  3. Fetch current portfolio and cash from the broker.
  4. Run the CVaR optimizer to produce rebalancing orders.
  5. Submit orders to the configured broker (paper or live Alpaca).
  6. Write an audit log entry for every order and fill.

Configuration is read from environment variables (see .env.example).
For local development, copy .env.example → .env and fill in values;
the script loads it automatically via python-dotenv.
"""

import datetime as dt
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add source/ to path before any local imports
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "source"))

# Load .env if present (python-dotenv; silently skipped if not installed)
try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging setup  (must happen before other imports so handlers are in place)
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Rotating daily log (keep 90 days)
    fh = logging.handlers.TimedRotatingFileHandler(
        log_dir / "portfolio.log",
        when="midnight",
        backupCount=90,
        utc=True,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Append-only audit log (never rotated — kept for 7+ years)
    audit_fh = logging.FileHandler(log_dir / "audit.log", mode="a")
    audit_fh.setFormatter(fmt)
    audit_logger = logging.getLogger("audit")
    audit_logger.addHandler(audit_fh)
    audit_logger.propagate = False  # audit entries only go to audit.log

    return logging.getLogger("run_daily")


# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    v = _env(key, str(default)).lower()
    return v in ("1", "true", "yes")


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Order submission with retry
# ---------------------------------------------------------------------------

def _submit_with_retry(broker, order, log, max_retries: int = 3):
    """Submit a single order, retrying up to *max_retries* times on failure."""
    for attempt in range(max_retries):
        try:
            fill = broker.submit_and_await_fill(order, timeout_seconds=60)
            return fill
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
            log.warning(
                "Order %s %s x%s failed (attempt %d/%d): %s — retrying in %ds",
                order.operation_type, order.ticker, order.qty,
                attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log_dir = Path(_env("LOG_DIR", "logs"))
    log = _setup_logging(log_dir, _env("LOG_LEVEL", "INFO"))
    audit = logging.getLogger("audit")

    today = dt.date.today()
    log.info("=== Daily job starting — %s ===", today.isoformat())

    # -- 1. NYSE trading-day check -------------------------------------------
    from market_hours import is_trading_day, IdempotencyGuard
    if not is_trading_day(today):
        log.info("Today (%s) is not a NYSE trading day. Exiting.", today.isoformat())
        return 0

    # -- 2. Idempotency guard -------------------------------------------------
    run_dir = Path(_env("RUN_STATE_DIR", "run_state"))
    guard = IdempotencyGuard(run_dir)

    if guard.already_ran(today):
        log.info("Job already completed successfully today (%s). Exiting.", today.isoformat())
        return 0

    if not guard.acquire_lock():
        log.error("Could not acquire run lock — another process may be running.")
        return 1

    try:
        return _run(log, audit, today)
    except Exception:
        log.exception("Unhandled exception in daily job.")
        return 2
    finally:
        guard.release_lock()
        log.info("=== Daily job finished — %s ===", today.isoformat())


def _run(log, audit, today: dt.date) -> int:
    from account_manager import rebalance_porfolio
    from market_hours import IdempotencyGuard

    # -- Config ---------------------------------------------------------------
    lookback_days  = _env_int("LOOKBACK_DAYS", 252)
    benchmark      = _env("BENCHMARK_SYMBOL", "SPY")
    fractional     = _env_bool("FRACTIONAL_SHARES", True)
    paper_mode     = _env_bool("ALPACA_PAPER", True)
    api_key        = _env("ALPACA_API_KEY")
    secret_key     = _env("ALPACA_SECRET_KEY")
    data_source    = _env("DATA_SOURCE", "local")
    data_file      = _env("DATA_FILE", "data/close.pkl")
    meta_file      = _env("METADATA_FILE", "data/metadata.pkl")
    cache_dir      = _env("DATA_CACHE_DIR", "data/cache/alpaca")
    run_dir        = Path(_env("RUN_STATE_DIR", "run_state"))
    budget         = _env_float("REBALANCE_BUDGET", 0.0)

    # -- 3. Build broker ------------------------------------------------------
    broker = _build_broker(api_key, secret_key, paper_mode, log)

    # -- 4. Fetch live portfolio and cash from broker -------------------------
    portfolio = broker.get_positions()
    cash = broker.get_cash()
    log.info(
        "Broker state: cash=%.2f, positions=%s",
        cash, list(portfolio.assets),
    )

    # Cap budget to available cash; if REBALANCE_BUDGET is 0 use all cash.
    deploy = min(budget, cash) if budget > 0 else cash
    if deploy <= 0:
        log.info("No cash available to deploy. Exiting.")
        _mark_success(run_dir, today, log)
        return 0
    log.info("Deploying up to $%.2f this rebalance.", deploy)

    # -- 5. Data manager ------------------------------------------------------
    try:
        if data_source == "alpaca":
            from data.alpaca import AlpacaDataManager
            data_manager = AlpacaDataManager(
                api_key=api_key,
                secret_key=secret_key,
                cache_dir=Path(cache_dir),
            )
            account_tickers = list(portfolio.assets) + [benchmark]
            data_manager.refresh_cache(account_tickers)
        else:
            from data.local import LocalDataManager
            data_manager = LocalDataManager(
                db_file=data_file,
                metadata_file=meta_file,
            )
    except Exception as exc:
        log.error("Failed to initialise data manager (%s): %s", data_source, exc)
        return 1

    # -- 6. Generate orders via CVaR optimizer --------------------------------
    end_date   = dt.datetime.combine(today, dt.time(16, 0))
    start_date = end_date - dt.timedelta(days=lookback_days)

    try:
        orders = rebalance_porfolio(
            portfolio=portfolio,
            additional_cash=deploy,
            start_date=start_date,
            end_date=end_date,
            data_manager=data_manager,
            print_portfolio=False,
            fractional_stocks=fractional,
        )
    except Exception as exc:
        log.error("Optimizer failed: %s", exc)
        return 1

    if not orders:
        log.info("Optimizer produced no orders — portfolio already optimal.")
        _mark_success(run_dir, today, log)
        return 0

    log.info("Optimizer produced %d orders.", len(orders))
    for o in orders:
        log.info("  ORDER  %s %s x%.4f @ %.4f", o.operation_type, o.ticker, o.qty, o.price)
        audit.info("ORDER  date=%s side=%s ticker=%s qty=%.4f price=%.4f",
                   today.isoformat(), o.operation_type, o.ticker, o.qty, o.price)

    # -- 7. Submit orders + collect fills -------------------------------------
    for order in orders:
        try:
            fill = _submit_with_retry(broker, order, log)
            if fill is None:
                log.warning("No fill received for %s %s — skipping.", order.operation_type, order.ticker)
                continue
            log.info(
                "  FILL   %s %s x%.4f @ %.4f (broker_id=%s)",
                fill.operation_type, fill.ticker, fill.qty,
                fill.fill_price, fill.broker_order_id,
            )
            audit.info(
                "FILL   date=%s side=%s ticker=%s qty=%.4f fill_price=%.4f broker_id=%s",
                today.isoformat(), fill.operation_type, fill.ticker,
                fill.qty, fill.fill_price, fill.broker_order_id,
            )
        except Exception as exc:
            log.error("Order %s %s failed after retries: %s", order.operation_type, order.ticker, exc)
            audit.error("ORDER_FAILED date=%s side=%s ticker=%s error=%s",
                        today.isoformat(), order.operation_type, order.ticker, exc)

    _mark_success(run_dir, today, log)
    return 0


def _build_broker(api_key, secret_key, paper_mode, log):
    """Return an AlpacaBroker if credentials are configured, else PaperBroker."""
    if api_key and secret_key:
        try:
            from broker.alpaca import AlpacaBroker
            broker = AlpacaBroker(api_key, secret_key, paper=paper_mode)
            mode = "paper" if paper_mode else "LIVE"
            log.info("Using AlpacaBroker (%s).", mode)
            return broker
        except Exception as exc:
            log.error("Failed to connect to Alpaca (%s). Falling back to PaperBroker.", exc)

    from broker.paper import CostAwarePaperBroker
    log.warning("No Alpaca credentials found — using local CostAwarePaperBroker (no real trades).")
    return CostAwarePaperBroker(cost_bps=5.0)


def _mark_success(run_dir, today, log):
    from market_hours import IdempotencyGuard
    IdempotencyGuard(run_dir).mark_success(today)
    log.info("Success marker written for %s.", today.isoformat())


if __name__ == "__main__":
    sys.exit(main())
