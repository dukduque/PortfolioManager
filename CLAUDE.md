# PortfolioManager — Developer Notes

User-facing documentation (setup, usage, deployment) is in [readme.md](readme.md).
This file contains only what is useful for editing the code.

---

## File layout

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

All settings live in `.env` (never committed). Full variable reference is in
[readme.md § Configure environment](readme.md#2-configure-environment).

Variables with non-obvious effects on the code:

| Variable | Default | Notes |
|---|---|---|
| `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` | | Blank → falls back to `CostAwarePaperBroker` |
| `REBALANCE_BUDGET` | `0` | 0 = all available cash; capped at `min(budget, cash)` in `run_daily.py` |
| `MAX_WEIGHT` | `1.0` | Passed as `max_weight` kwarg to `rebalance_porfolio()` |
| `LOOKBACK_DAYS` | `730` | Calendar days — 730 ≈ 2 trading years of returns |
| `DATA_SOURCE` | `alpaca` | `local` only for offline backtesting with pickle files |

---

## Broker interface

Two ABCs in `source/broker/base.py`:

- **`BrokerBase`** — operational path (`run_daily.py`): `get_positions()`, `get_cash()`,
  `submit_order()`, `submit_and_await_fill()`, `cancel_order()`, `is_market_open()`
- **`BrokerHistory`** — analytics path (`account_from_broker()`): `get_order_history()`,
  `get_cash_flows()`, `supports_cash_flow_history`

When adding a new broker implement `BrokerBase` at minimum. Add `BrokerHistory` and
set `supports_cash_flow_history = True` only if the broker exposes transaction history
(required for equity curve and benchmark comparison).

---

## CVaR optimizer

`source/opt_tools.py` — `cvar_model_ortools` builds the LP/MIP once. Call
`change_cvar_params(alpha, beta)` to re-solve cheaply (only the objective is rebuilt).

- `alpha` (default 0.90): CVaR confidence level — covers worst (1−α) fraction of scenarios
- `beta` (default 0.95): return vs risk weight; 0 = pure CVaR min, 1 = pure return max
- `max_weight` (default 1.0): per-asset allocation cap. Without this, LP corner solutions
  concentrate everything in the single highest-mean asset.
- `portfolio_delta` (default 1e9): turnover limit in dollars; 1e9 = sells freely allowed

`rebalance_porfolio()` in `account_manager.py` is the public entry point; it reads
`max_weight` and `portfolio_delta` from `**kwargs`.

---

## Entry point flags

```shell
python run_daily.py --dry-run    # skip guards + skip order submission (safe any time)
python run_daily.py --no-guard   # skip guards, place real orders
python run_daily.py              # production: guards active, places real orders
```

Guards = NYSE trading-day check + idempotency lock (prevents double-runs on the same day).

---

## S&P 500 universe

`database_handler.save_sp500_tickers()` scrapes Wikipedia and caches the result at
`data/sp500tickers.pickle` (gitignored, regenerated on first run). `_load_universe()`
in `run_daily.py` loads this cache; on failure it falls back to a hardcoded 15-ticker list.

---

## Logging

All modules use Python `logging` — never `print`. `run_daily.py` configures:
- `logs/portfolio.log` — rotating daily, 90-day retention
- `logs/audit.log` — append-only, never rotated

`rebalance_porfolio()` always logs the portfolio table and CVaR stats regardless of
the `print_portfolio` flag (that flag only controls per-order log lines).

---

## Tests

```shell
pytest
```

`tests/fake.py` provides `FakeBroker` and `FakeDataManager`. New tests should use
these fakes and live under `tests/`.
