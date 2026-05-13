# PortfolioManager — Developer Notes

Full user-facing documentation is in [readme.md](readme.md). This file covers
what Claude needs to know to work effectively in this codebase.

---

## What this does

Daily portfolio rebalancing via a mean-CVaR optimizer (OR-Tools CBC) connected
to the Alpaca Trading API. See [readme.md § How it works](readme.md#how-it-works).

---

## Layout

```
run_daily.py               Entry point: guards → broker → data → optimize → orders
source/
  account_manager.py       rebalance_porfolio(): data prep + CVaR solve + order gen
  opt_tools.py             cvar_model_ortools: mean-CVaR MIP (OR-Tools CBC)
  resources.py             Portfolio, Order, Fill, CashFlow; account_from_broker()
  database_handler.py      S&P 500 universe fetch (Wikipedia scrape, cached locally)
  backtest.py              build_equity_curve(), build_benchmark_curve()
  market_hours.py          NYSE calendar checks; IdempotencyGuard
  broker/
    base.py                BrokerBase (operational) + BrokerHistory (analytics) ABCs
    alpaca.py              AlpacaBroker: live/paper + order history + cash flows
    paper.py               CostAwarePaperBroker: in-memory simulation
  data/
    alpaca.py              AlpacaDataManager: incremental parquet price cache
    local.py               LocalDataManager: legacy pickle-based source
deploy/
  portfolio-manager.service  systemd service unit
  portfolio-manager.timer    systemd timer (Mon-Fri 15:00 UTC = 10:00 ET)
tests/                     pytest suite
```

---

## Configuration

All settings live in `.env` (copied from `.env.example`; never committed).
See [readme.md § Configure environment](readme.md#2-configure-environment) for
the full variable reference.

Key variables Claude should be aware of when editing code:

| Variable | Default | Notes |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | | Blank = fall back to `CostAwarePaperBroker` |
| `ALPACA_PAPER` | `true` | Must be `true` unless intentionally going live |
| `REBALANCE_BUDGET` | `0` | 0 = deploy all available cash |
| `LOOKBACK_DAYS` | `730` | Calendar days — 730 ≈ 2 trading years |
| `MAX_WEIGHT` | `1.0` | Per-asset cap; `0.05` = equal-weight over 20 stocks |
| `DATA_SOURCE` | `alpaca` | `local` only for offline backtesting |

---

## Broker interface

Split into two ABCs (see `source/broker/base.py`):

- **`BrokerBase`** — operational path used by `run_daily.py`: `get_positions()`,
  `get_cash()`, `submit_order()`, `submit_and_await_fill()`, `cancel_order()`,
  `is_market_open()`
- **`BrokerHistory`** — analytics path used by `account_from_broker()`:
  `get_order_history()`, `get_cash_flows()`, `supports_cash_flow_history`

When adding a new broker, implement `BrokerBase` at minimum. Implement
`BrokerHistory` and set `supports_cash_flow_history = True` if the broker
exposes transaction history (needed for equity curve / benchmark comparison).

---

## CVaR optimizer

`source/opt_tools.py` — `cvar_model_ortools` builds the LP/MIP once; call
`change_cvar_params(alpha, beta)` to re-solve cheaply (objective only rebuilt).

Key parameters:
- `alpha` (default 0.90): CVaR confidence level
- `beta` (default 0.95): return vs risk weight; 0 = pure CVaR min, 1 = pure return max
- `max_weight` (default 1.0): per-asset allocation cap — the most effective
  lever for forcing diversification; LP corner solutions concentrate everything
  in the top-mean asset without this

`rebalance_porfolio()` in `account_manager.py` is the public entry point.
It reads `max_weight` and `portfolio_delta` from `**kwargs`, defaulting to
`1.0` and `1e9` (unconstrained, sells allowed) respectively.

---

## Entry point flags

```shell
python run_daily.py --dry-run    # skip guards + skip order submission (safe any time)
python run_daily.py --no-guard   # skip guards, place real orders
python run_daily.py              # production mode: guards active, places real orders
```

`test_rebalance.py` is a thin shim that forces `--dry-run` and delegates to
`run_daily.main()`. Use `run_daily.py --dry-run` directly — they are identical.

---

## S&P 500 universe

`database_handler.save_sp500_tickers()` scrapes Wikipedia and caches the result
at `data/sp500tickers.pickle`. `_load_universe()` in `run_daily.py` loads this
cache; if it fails it falls back to a hardcoded 15-ticker list. The pickle is
gitignored — it is regenerated on first run.

---

## Logging

All modules use Python `logging` (never `print`). `run_daily.py` sets up:
- `logs/portfolio.log` — rotating daily, 90-day retention
- `logs/audit.log` — append-only, never rotated (order and fill records)

The portfolio table and CVaR stats are always logged by `rebalance_porfolio()`
regardless of the `print_portfolio` flag (which controls per-order lines only).

---

## Tests

```shell
pytest
```

See `tests/` for the suite. Mocking strategy: `tests/fake.py` provides
`FakeBroker` and `FakeDataManager`. New tests should follow this pattern and
live under `tests/`.

---

## Deployment

See [readme.md § Deployment](readme.md#deployment) for full instructions
covering local cron, VPS systemd, and Docker.

Quick reference:
- Systemd units: `deploy/portfolio-manager.{service,timer}`
- Secrets on VPS: `/etc/portfolio-manager/secrets.env` (chmod 600)
- Docker: `docker compose run --rm portfolio-manager`
