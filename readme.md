## <span style="color:red"> Disclaimer: NOT FINANCIAL ADVICE </span>

**This software is for informational and educational purposes only and does not constitute or intend to be financial advice. Investing is risky, do your own research and consult a financial advisor before making any investment decision.**

# Portfolio Manager

| **Build Status** | **Coverage** |
|:--------------------:|:----------------:|
|[![Build Status][build-img]][build-url] | [![Codecov branch][codecov-img]][codecov-url]

[codecov-img]: https://codecov.io/github/dukduque/PortfolioManager/coverage.svg?branch=master
[codecov-url]: https://codecov.io/github/dukduque/PortfolioManager?branch=master

[build-img]: https://github.com/dukduque/PortfolioManager/workflows/CI/badge.svg?branch=master
[build-url]: https://github.com/dukduque/PortfolioManager/actions?query=workflow%3ACI

A daily portfolio rebalancing system backed by a mean-CVaR optimizer (OR-Tools) and the Alpaca Trading API. Each day it fetches live positions and cash from Alpaca, runs the optimizer over the S&P 500 universe, and places market orders automatically.

## How it works

1. Fetches current positions and cash from Alpaca (paper or live).
2. Refreshes a local parquet price cache for the full S&P 500 universe.
3. Runs a mean-CVaR mixed-integer programme (OR-Tools CBC) to find the optimal portfolio allocation given the available budget.
4. Submits market orders to Alpaca and writes an audit log.

---

## Setup

### 1. Install dependencies

```shell
pip install -r requirements.txt
```

### 2. Configure environment

```shell
cp .env.example .env
```

**Where `.env` lives:** it must be placed in the project root (same directory as `run_daily.py`). The scripts load it automatically at startup via `python-dotenv`.

**Why `.env` is not in the repository:** it contains your Alpaca API credentials. Committing secrets to version control is a security risk — even in private repositories. The file is listed in `.gitignore` to make accidental commits impossible. `.env.example` is committed instead and documents every supported variable with safe defaults.

Edit `.env` with your values:

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | | Alpaca API key — get one at [app.alpaca.markets](https://app.alpaca.markets) |
| `ALPACA_SECRET_KEY` | | Alpaca secret key |
| `ALPACA_PAPER` | `true` | `true` = paper account; `false` = live trading (use with care) |
| `REBALANCE_BUDGET` | `0` | Max USD to deploy per run; `0` deploys all available cash |
| `LOOKBACK_DAYS` | `730` | Calendar days of return history for the CVaR optimizer (~2 trading years) |
| `MAX_WEIGHT` | `1.0` | Max fraction of budget in any single asset; `0.25` caps positions at 25% |
| `FRACTIONAL_SHARES` | `true` | Allow fractional share quantities (Alpaca supports this) |
| `BENCHMARK_SYMBOL` | `SPY` | Passive benchmark ticker for performance reports |
| `DATA_SOURCE` | `alpaca` | `alpaca` (recommended) or `local` (legacy pickle files) |
| `DATA_CACHE_DIR` | `./data/cache/alpaca` | Parquet cache directory for Alpaca price data |
| `LOG_DIR` | `./logs` | Directory for rotating logs and the audit log |
| `LOG_LEVEL` | `INFO` | Python log level: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### 3. PYTHONPATH

`run_daily.py`, `test_rebalance.py`, and `sweep_cvar.py` all prepend `source/` to `sys.path` at startup, so **no manual configuration is needed** when running those scripts directly.

If you import project modules from outside those scripts (e.g. a REPL, a notebook, or an IDE run configuration), set `PYTHONPATH` for your platform:

**macOS / Linux**
```shell
export PYTHONPATH=/path/to/PortfolioManager/source
# Add to ~/.zshrc or ~/.bashrc to make it permanent
```

**Windows — Command Prompt**
```cmd
set PYTHONPATH=C:\path\to\PortfolioManager\source
```

**Windows — PowerShell**
```powershell
$env:PYTHONPATH = "C:\path\to\PortfolioManager\source"
# Add to $PROFILE to make it permanent
```

**Windows — permanent (System Environment Variables)**

Open *Settings → System → About → Advanced system settings → Environment Variables*, add a User variable `PYTHONPATH` with value `C:\path\to\PortfolioManager\source`.

---

## Usage

### Dry run — preview orders, no trades placed

```shell
python run_daily.py --dry-run
```

Bypasses the NYSE trading-day check and idempotency guard. Prints the proposed portfolio table and orders but does not submit anything to Alpaca. Safe to run any time, any day.

### Manual run — place real orders now

```shell
python run_daily.py --no-guard
```

Skips the trading-day and idempotency checks but submits real orders to Alpaca. Useful for ad-hoc or forced manual rebalances outside the normal schedule.

### Scheduled daily run

```shell
python run_daily.py
```

Exits cleanly if today is not a NYSE trading day or if the job already ran successfully today. Designed to be called by a scheduler (see [Deployment](#deployment) below).

### Parameter sensitivity sweep

```shell
python sweep_cvar.py --budget 10000 --max-weight 0.25
```

Runs the optimizer across a grid of CVaR α and β values and prints a table of expected return, CVaR, VaR, and portfolio composition for each combination. Always a dry run — no orders are submitted.

---

## Architecture

```
run_daily.py                  # Entry point — scheduling guards, broker wiring, order submission
source/
  account_manager.py          # rebalance_porfolio(): data prep + CVaR solve + order generation
  opt_tools.py                # cvar_model_ortools: mean-CVaR MIP (OR-Tools CBC)
  resources.py                # Portfolio, Order, Fill, CashFlow dataclasses; account_from_broker()
  database_handler.py         # S&P 500 universe fetch (Wikipedia) + local pickle cache
  backtest.py                 # build_equity_curve(), build_benchmark_curve()
  broker/
    base.py                   # BrokerBase (operational) + BrokerHistory (analytics) ABCs
    alpaca.py                 # AlpacaBroker: live/paper trading + order/cash-flow history
    paper.py                  # CostAwarePaperBroker: local simulation with transaction costs
  data/
    alpaca.py                 # AlpacaDataManager: incremental parquet price cache
    local.py                  # LocalDataManager: legacy pickle-based data source
deploy/
  portfolio-manager.service   # systemd service unit
  portfolio-manager.timer     # systemd timer (fires Mon-Fri 10:00 ET)
```

### Broker interface

The broker layer is split into two abstract base classes:

- **`BrokerBase`** — operational: `get_positions()`, `get_cash()`, `submit_order()`, `submit_and_await_fill()`
- **`BrokerHistory`** — analytics: `get_order_history()`, `get_cash_flows()`

`AlpacaBroker` implements both. `CostAwarePaperBroker` implements both with an in-memory ledger. Leaving `ALPACA_API_KEY` blank falls back automatically to `CostAwarePaperBroker` — no real trades, no account needed.

### CVaR optimizer

The optimizer solves a mean-CVaR mixed-integer programme at every rebalance:

```
max  β · E[return]  −  (1−β) · CVaR_α(loss)

s.t. Σ p_j x_j + cash = B          (budget)
     p_j x_j / B ≤ max_weight       (position cap — breaks LP corner solutions)
     x_j ≥ 0                        (no short selling)
```

| Parameter | Default | Description |
|---|---|---|
| α | 0.90 | CVaR confidence level — covers worst (1−α) fraction of scenarios |
| β | 0.95 | Return weight — 0 = pure risk minimisation, 1 = pure return maximisation |
| `MAX_WEIGHT` | 1.0 | Per-asset cap — set to e.g. 0.05 for equal-weight-style diversification |

---

## Deployment

### Option 1 — Local machine (cron)

Run the job locally on any machine that is on during market hours. Add a crontab entry to fire at 10:00 ET (15:00 UTC) on weekdays:

```shell
crontab -e
```

```cron
0 15 * * 1-5 cd /path/to/PortfolioManager && /usr/bin/python run_daily.py >> logs/cron.log 2>&1
```

The job exits immediately on non-trading days, so the cron schedule does not need to know about NYSE holidays.

### Option 2 — VPS with systemd (recommended)

Pre-built systemd unit files are included in `deploy/`. This is the most robust option — systemd restarts on transient failures, logs to journald, and runs as a dedicated non-root user.

```shell
# 1. Copy the app to the server
scp -r . user@your-server:/opt/portfolio-manager

# 2. Create a dedicated non-root user
sudo useradd -r -s /sbin/nologin portfolio

# 3. Create the secrets file (mode 600, owned by portfolio)
sudo mkdir -p /etc/portfolio-manager
sudo cp .env /etc/portfolio-manager/secrets.env
sudo chown portfolio:portfolio /etc/portfolio-manager/secrets.env
sudo chmod 600 /etc/portfolio-manager/secrets.env

# 4. Install Python dependencies
sudo -u portfolio python -m venv /opt/portfolio-manager/venv
sudo -u portfolio /opt/portfolio-manager/venv/bin/pip install -r requirements.txt

# 5. Install and enable the systemd units
sudo cp deploy/portfolio-manager.service /etc/systemd/system/
sudo cp deploy/portfolio-manager.timer   /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now portfolio-manager.timer

# 6. Verify
sudo systemctl status portfolio-manager.timer
journalctl -u portfolio-manager.service -f
```

The timer fires Mon–Fri at 15:00 UTC (10:00 ET). `run_daily.py` handles NYSE holidays itself, so the timer does not need to be aware of them.

### Option 3 — Docker

No pre-built image is published. Build and run locally or on a VPS:

**`Dockerfile`:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_daily.py"]
```

**`docker-compose.yml`:**

```yaml
services:
  portfolio-manager:
    build: .
    env_file:
      - .env          # never commit this file
    volumes:
      - ./data:/app/data           # persist price cache across runs
      - ./logs:/app/logs           # persist audit log
      - ./run_state:/app/run_state # persist idempotency markers
    restart: "no"     # scheduler triggers a fresh container each run
```

Run manually:

```shell
docker compose run --rm portfolio-manager
```

Schedule with cron on the host (fires Mon–Fri at 10:00 ET):

```cron
0 15 * * 1-5 cd /path/to/PortfolioManager && docker compose run --rm portfolio-manager >> logs/docker-cron.log 2>&1
```

> **Secrets:** never bake `.env` into the image. Pass it via `env_file` in Compose or as `--env-file` in `docker run`. On a VPS, store secrets in `/etc/portfolio-manager/secrets.env` (chmod 600) and mount or reference from there.

---

## Logs

| File | Description |
|---|---|
| `logs/portfolio.log` | Rotating daily log, kept 90 days |
| `logs/audit.log` | Append-only order and fill record, never rotated |

---

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned features including a web configuration panel, analytics dashboard, multi-broker support, and alerting.
