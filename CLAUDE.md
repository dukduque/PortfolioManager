# PortfolioManager — Developer Notes

## What this does

A Python tool for building and rebalancing an equity portfolio using CVaR (Conditional Value-at-Risk) optimization. Price data is fetched from Alpaca or local pickle files. The daily job (`run_daily.py`) updates data, computes optimal weights, and places orders via a pluggable broker interface.

## Layout

```
run_daily.py               Daily job entry point: data update → optimize → rebalance
source/
  resources.py             Account, Order, position data structures; load/save account
  account_manager.py       Rebalancing logic, pie-chart reporting
  database_handler.py      Price-data cache: fetch, update, persist (Alpaca or local)
  opt_tools.py             CVaR optimization (scipy/cvxpy)
  backtest.py              Backtesting harness
  market_hours.py          Market-hours and holiday checks
  broker/                  Generic broker interface; Alpaca implementation + PaperBroker stub
  util.py                  Small utilities
data/                      Price pickle files (legacy local mode)
deploy/                    Deployment configs
examples/                  Usage examples
tests/                     pytest test suite
DEPLOYMENT_PLAN.md         Production deployment guide (broker integration, scheduling, security)
.env.example               Full env var documentation — copy to .env before running
```

## Configuration

All runtime settings live in `.env` (copy `.env.example`). Key variables:

| Variable           | Default        | Purpose                                              |
|--------------------|----------------|------------------------------------------------------|
| ALPACA_API_KEY     | —              | Alpaca credentials (blank = PaperBroker, no trades)  |
| ALPACA_SECRET_KEY  | —              |                                                      |
| ALPACA_PAPER       | true           | Set to `false` for live trading                      |
| ACCOUNT_NAME       | my_portfolio   | Account folder name under ACCOUNT_PATH               |
| LOOKBACK_DAYS      | 252            | Calendar days of return history for CVaR optimizer   |
| BENCHMARK_SYMBOL   | SPY            | Passive benchmark ticker                             |
| DATA_SOURCE        | alpaca         | `alpaca` (recommended) or `local` (legacy pickles)   |
| LOG_LEVEL          | INFO           | Python log level: DEBUG / INFO / WARNING / ERROR     |

## Broker interface

`source/broker/` defines a generic account interface. The Alpaca implementation is included. Leaving `ALPACA_API_KEY` blank falls back to `PaperBroker` (simulates orders locally, no real trades). Implement the interface to connect any other broker.

## PYTHONPATH

Must include `source/` for all imports to resolve:

```bash
export PYTHONPATH=path/to/PortfolioManager/source
```

## Logging

All modules use Python's `logging` (not `print`). The root logger is configured in `run_daily.py` with a `StreamHandler`. Running individual modules directly (e.g. `python database_handler.py`) also has a `basicConfig` guard for CLI output.

## Running tests

```bash
pytest
```

## Update price data

```bash
python source/database_handler.py -a=u -db_file="close.pkl" -days_back=3 -n_proc=4
```
