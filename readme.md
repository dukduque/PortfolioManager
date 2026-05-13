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

`run_daily.py` prepends `source/` to `sys.path` at startup, so **no manual configuration is needed** when running it directly.

If you import project modules from outside that script (e.g. a REPL, a notebook, or an IDE run configuration), add `source/` to `PYTHONPATH`. The commands below **append** to any existing value so nothing already on the path is lost.

**macOS / Linux**
```shell
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/path/to/PortfolioManager/source"
# Add to ~/.zshrc or ~/.bashrc to make it permanent
```

**Windows — Command Prompt**
```cmd
set PYTHONPATH=%PYTHONPATH%;C:\path\to\PortfolioManager\source
```

**Windows — PowerShell**
```powershell
$env:PYTHONPATH = ($env:PYTHONPATH, "C:\path\to\PortfolioManager\source" | Where-Object { $_ }) -join ";"
# Add to $PROFILE to make it permanent
```

**Windows — permanent (System Environment Variables)**

Open *Settings → System → About → Advanced system settings → Environment Variables*. If `PYTHONPATH` already exists, **edit** it and append `;C:\path\to\PortfolioManager\source` to the existing value. If it does not exist, create a new User variable named `PYTHONPATH` with value `C:\path\to\PortfolioManager\source`.

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

The optimizer solves a mean-CVaR mixed-integer programme (Rockafellar & Uryasev, 2000) at every rebalance. The inputs are *n* historical return scenarios for *m* assets.

**Notation**

| Symbol | Meaning |
|---|---|
| n, m | number of historical scenarios; number of assets |
| r_{ij} | gross return (price ratio) of asset j in scenario i: r_{ij} = price_{j,t+1} / price_{j,t} |
| r̄_j | sample mean gross return of asset j across all scenarios |
| p_j | current price of asset j |
| B | total budget = current portfolio equity (mark-to-market) + available cash |
| x_j ≥ 0 | **decision variable** — shares held in asset j |
| cash ≥ 0 | **decision variable** — uninvested cash (earns zero return) |
| η | **decision variable** — Value at Risk threshold (the α-quantile of the loss distribution) |
| z_i ≥ 0 | **decision variable** — per-scenario shortfall above η (CVaR auxiliary variable) |
| α | CVaR confidence level — the CVaR covers the worst (1−α) fraction of scenarios |
| β | convex weight on expected return vs CVaR in the objective |

**Portfolio return and loss**

The portfolio gross return in scenario i is the dollar-weighted average return:

```
ρ_i = ( Σ_j r_{ij} · p_j · x_j  +  cash ) / B
```

The gross loss is simply the negative of this: L_i = −ρ_i.

**Programme**

CVaR is linearised via the Rockafellar–Uryasev auxiliary variable z_i ≥ (L_i − η)⁺:

```
max   β · Σ_j r̄_j · p_j · x_j / B
    − (1−β) · [ η + 1/(n·(1−α)) · Σ_i z_i ]

s.t.  Σ_j p_j · x_j + cash = B                          (budget)
      z_i ≥ −Σ_j r_{ij} · p_j · x_j / B − cash/B − η   (CVaR linearisation, ∀i)
      z_i ≥ 0                                            (∀i)
      p_j · x_j / B ≤ max_weight                         (position cap)
      x_j ≥ 0, cash ≥ 0                                  (no short selling)
```

The position cap `p_j · x_j / B ≤ max_weight` is the most important practical lever: without it, the linear objective always selects a corner solution that concentrates the entire budget in a single asset.

**Parameters**

| Parameter | Default | Description |
|---|---|---|
| α | 0.90 | CVaR confidence level — e.g. 0.90 means the CVaR covers the worst 10 % of scenarios |
| β | 0.95 | Return weight — 0 = minimise CVaR only, 1 = maximise expected return only |
| `MAX_WEIGHT` | 1.0 | Per-asset cap as a fraction of B — e.g. 0.05 forces at least 20 positions |

---

## Deployment

### Option 1 — Local machine

Run the job on any machine that is on and connected to the internet at 10:00 ET on weekdays. The machine does not need to stay on all day — only during the rebalance window.

**Prerequisites**
- `.env` configured in the project root
- Python and dependencies installed (`pip install -r requirements.txt`)
- You know the exact path to your Python executable (see below for how to find it)

#### macOS and Linux

**What is cron?** `cron` is the built-in Unix/macOS job scheduler. It runs commands on a schedule defined in a file called the crontab.

**Is cron installed?** On macOS it is always available. On Linux it depends on the distribution — check with:

```shell
# Debian / Ubuntu
systemctl status cron

# RHEL / Fedora / CentOS
systemctl status crond
```

If the service is not found, install it:

```shell
# Debian / Ubuntu
sudo apt-get install cron && sudo systemctl enable --now cron

# RHEL / Fedora
sudo dnf install cronie && sudo systemctl enable --now crond
```

**Find your Python path** — the correct path varies by installation method:

```shell
# System Python or pyenv
which python3

# Inside a virtual environment (activate it first)
source /path/to/venv/bin/activate && which python
```

**Add the cron entry:**

```shell
crontab -e
```

Add this line, replacing the paths with your own:

```cron
0 15 * * 1-5 cd /path/to/PortfolioManager && /path/to/python run_daily.py >> logs/cron.log 2>&1
```

- `0 15 * * 1-5` — fire at 15:00 UTC (10:00 ET) Monday through Friday
- `>> logs/cron.log` — append stdout to a log file; create `logs/` first if it does not exist
- `2>&1` — redirect stderr into the same log file so errors are captured

The `cd` is required because `run_daily.py` resolves all paths (`.env`, cache, logs) relative to its working directory.

> **Note:** cron runs with a minimal environment — no shell profile, no conda/pyenv shims. Always use the absolute path to the Python executable, not `python` or `python3`.

The job exits immediately on non-trading days, so the cron schedule does not need to know about NYSE holidays.

#### Windows

Windows does not have cron. The equivalent is **Task Scheduler**, which is built into every version of Windows.

**Find your Python path:**

```powershell
(Get-Command python).Source
```

If you are using a virtual environment, activate it first, then run the command above.

**Create the scheduled task:**

1. Open **Task Scheduler** (search for it in the Start menu).
2. In the right-hand panel click **Create Task** (not "Create Basic Task" — Basic Task does not let you set the timezone or weekday filter precisely).
3. **General tab:**
   - Name: `PortfolioManager daily rebalance`
   - Select *Run only when user is logged on* (simplest) or *Run whether user is logged on or not* (runs even with the screen locked, but requires your Windows password).
   - Check *Run with highest privileges* if your Python installation requires it.
4. **Triggers tab → New:**
   - Begin the task: *On a schedule*
   - Settings: *Weekly*, repeat every 1 week, check Mon–Fri
   - Start: set the time to **10:00 AM** and make sure the timezone shown matches Eastern Time. If your machine is set to a different timezone, convert accordingly (e.g. 9:00 AM Central, 7:00 AM Pacific).
   - Check *Enabled*.
5. **Actions tab → New:**
   - Action: *Start a program*
   - Program/script: full path to `python.exe`, e.g. `C:\Users\YourName\AppData\Local\Programs\Python\Python312\python.exe`
   - Add arguments: `run_daily.py`
   - Start in: full path to the project root, e.g. `C:\Users\YourName\PortfolioManager`
6. **Conditions tab:**
   - Uncheck *Stop if the computer switches to battery power* if you are on a laptop and want it to run on battery.
   - If you want the machine to wake from sleep to run the job, check *Wake the computer to run this task* — note this requires the machine to be in sleep (not hibernation or shut down).
7. **Settings tab:**
   - Check *Run task as soon as possible after a scheduled start is missed* — this handles the case where the machine was off or asleep at trigger time.
8. Click **OK**.

**Verify it works** by right-clicking the task and selecting *Run*. Check `logs\portfolio.log` and `logs\audit.log` to confirm the job executed.

> **Tip:** if the task silently does nothing, open **Event Viewer → Windows Logs → Application** and filter by source `PortfolioManager` or look for errors around the trigger time. A common cause is an incorrect *Start in* path — without it Python cannot find `run_daily.py` or `.env`.

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

