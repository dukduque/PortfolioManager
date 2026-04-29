# PortfolioManager: Production Deployment Plan

## Context

The PortfolioManager is a CVaR-based portfolio optimizer that generates buy/sell orders but has no automated execution, no broker integration, no scheduling, and no monitoring. The goal is to deploy it to run daily and place real trades automatically. The plan covers backtesting validation, broker selection, deployment architecture, scheduling, security, and monitoring.

---

## 1. Backtesting

### Current Gaps
- `benchmark()` / `build_account_history()` are performance *measurement*, not true backtesting — they evaluate already-executed portfolios, not historical decisions.
- `DataManager.get_returns()` caches `self._returns` on first call regardless of date args — silently reuses stale data in any walk-forward loop.
- `update_database()` drops columns with all-NaN (delisted stocks removed) = survivorship bias.
- No transaction cost modeling.

### Walk-Forward Design
```
for rebalance_date in backtest_calendar:
    train_window = [rebalance_date - lookback, rebalance_date]
    1. Fetch prices strictly as-of rebalance_date
    2. Run cvar_model_ortools on those returns
    3. Apply transaction cost haircut to orders
    4. Update simulated portfolio
    5. Record portfolio value
```
Key fix: pass `end_date=rebalance_date` strictly; fix `_returns` caching bug first.

### Transaction Costs to Model
| Type | Approach |
|---|---|
| Commission | $0 Alpaca, $0.005/share IBKR |
| Bid-ask spread | `fill_price = mid * (1 ± spread/2)`; ~5bps for large-cap |
| Slippage | 0.1–0.2% on order size (negligible for <$500K) |

### Risk Metrics to Add
- Sharpe, Sortino, Calmar (annualized return / max drawdown)
- Max drawdown, rolling beta vs SPY
- Turnover (fraction of portfolio replaced per rebalance)
- Win rate vs benchmark

### Recommended Libraries
- **VectorBT** — walk-forward simulation engine, numpy-native, fast
- **QuantStats** — HTML tearsheet generation from equity curve
- `pandas-market-calendars` — NYSE holiday schedule for both backtester and scheduler

### Data Quality Fixes
- Survivorship bias: maintain `constituents.pkl` mapping `(date, index) → [tickers]`; use period-accurate universe per rebalance date
- Split/dividend consistency: verify `close.pkl` uses adjusted close throughout
- Missing data: forward-fill up to 5 days, then drop (current `dropna(axis=1)` is too aggressive)
- Holiday NaN rows: replace weekday-only date filtering with NYSE calendar

---

## 2. Broker Options

### Alpaca ✅ Recommended
- Commission-free; paper trading environment with separate keys
- Fractional shares on market orders (aligns with `fractional=True` flag)
- Good REST API; `alpaca-py` is the current maintained SDK
- **Action**: Remove deprecated `alpaca_trade_api==1.2.3`; add `alpaca-py>=0.18.0`
- IP allowlisting supported — use it

### Interactive Brokers
- Broader instruments (options, futures, international)
- Requires TWS or IB Gateway running as a local process — complicates cloud deployment
- Python: `ib_insync`
- Not recommended unless you need instruments beyond US equities/ETFs

### Schwab (post-TD Ameritrade)
- No paper trading environment; no fractional shares — hard blockers for this project

### Tradier
- No fractional shares; no official Python SDK — inferior to Alpaca for this use case

---

## 3. Deployment Architecture

### Option A: Cloud VPS — ✅ Recommended for solo developer
- DigitalOcean / Hetzner / Linode: ~$6–12/month
- Always-on, persistent disk, easy to SSH/debug
- systemd timer for scheduling, environment variables for secrets
- Add nightly backup to cloud storage (~$1–2/month)

### Option B: GCP Cloud Run Jobs + Cloud Scheduler
- Containerized Python task; ~$0.006/run; Cloud Scheduler: $0.10/month
- Good fit if already using GCP; cleaner for ops at scale
- Slightly higher setup complexity (Docker, Artifact Registry, Service Account)

### Option C: AWS EC2 t3.micro / Fargate
- EC2: ~$8–12/month on-demand; similar to VPS with better native AWS integration
- Fargate: elegant for scheduled workloads; 30–90s cold start acceptable for daily runs
- OR-Tools (~100MB compiled binary) likely exceeds Lambda's 250MB limit → avoid Lambda

### Option D: GitHub Actions scheduled workflows
- Free tier covers ~220 min/month (fits within limit)
- **Not recommended for real money**: timing unreliability (up to 30min queue delay), no persistent state between runs
- Fine for backtesting CI jobs and paper trading sanity checks

---

## 4. Scheduling Strategy

### Trigger Time
- **10:00 ET** (30 min after open): avoids opening volatility, still within trading window
- Cron UTC: `0 15 * * 1-5` (EDT) / `0 14 * * 1-5` (EST)
- **Better**: Start script with a `pandas_market_calendars` NYSE check; exit cleanly on holidays/early close

### Idempotency (Critical — currently absent)
1. Write a run-lock record at job start; check it on re-entry; write success marker at end
2. Before submitting orders, query broker for existing positions; skip orders already filled
3. Pass `client_order_id = f"{account}_{date}_{ticker}_{side}"` to Alpaca — duplicate rejected with 422

### Retry Logic
- API-level: already in `update_stock_prices()` (3 retries); extend to order submission
- Job-level (systemd): `Restart=on-failure`, `RestartSec=900`, `StartLimitBurst=3`

---

## 5. Security

### Secrets Management
- **Development**: `.env` + `python-dotenv`; must be in `.gitignore`
- **VPS production**: systemd `EnvironmentFile=/etc/portfolio-manager/secrets.env` with `chmod 600`, owned by service user
- **Cloud**: AWS Secrets Manager or GCP Secret Manager

### API Key Practices
- Separate paper vs. live keys; never use live keys in dev/test
- Alpaca: use minimal permissions (`orders:write`, `positions:read`, `account:read`)
- Enable IP allowlisting on Alpaca dashboard — only the VPS IP can trade
- Rotate keys monthly; revoke old key after new one confirmed working

### State File Protection
- Add `*.pkl`, `*.acc`, `data/`, `accounts/` to `.gitignore` immediately
- Encrypt data directory (LUKS on VPS, or SSE on S3/GCS)
- Daily backup to a separate cloud storage bucket
- Use atomic writes: write temp file → rename (avoids corrupt state on crash)

### Audit Trail
- Log every order submission and every fill confirmation to an append-only file
- Include: ticker, side, qty, intended price, fill price, fill time, broker order ID
- Retain ≥7 years (IRS/FINRA personal trading record requirements)

---

## 6. Monitoring and Alerting

### Logging
Replace all `print()` statements with structured JSON logging:
- **INFO**: job start/end, each order, each fill, portfolio snapshot, CVaR stats
- **WARNING**: stale prices, optimizer gap > threshold, partial fills
- **ERROR**: API failures, optimizer infeasible
- **CRITICAL**: broker rejection, state inconsistency

Start with rotating file logs on VPS. Add Loki + Grafana (self-hosted, ~200MB RAM) for dashboards when Phase 2 is stable.

### Alerting
1. **Email** (Phase 2): `smtplib` or SendGrid free tier; send on ERROR/CRITICAL
2. **Slack webhook** (Phase 2): formatted message on failure
3. **SMS via Twilio** (Phase 3): for CRITICAL events only (~$0.008/SMS)

### Portfolio Drift Monitoring
- Daily check even on non-rebalance days
- Alert if any position drifts >5% absolute from target weights

### Daily Report (automated email after execution)
- Portfolio value: today vs. yesterday vs. 30 days ago
- SPY benchmark comparison
- Orders placed with fill prices
- Running Sharpe, max drawdown, Calmar
- CVaR stats for new portfolio
- `benchmark2()` in `account_manager.py` already generates most of this; remove `plt.show()`, save figures to file

---

## 7. Phased Implementation Plan

### Phase 1: Backtesting + Paper Trading (4–8 weeks)

**Week 1–2: Fix Data Quality**
- Fix `DataManager._returns` caching bug (stale data in walk-forward loops)
- Add `pandas-market-calendars` for NYSE holiday handling in `update_database()`
- Add `*.pkl`, `*.acc` to `.gitignore`
- Fix pandas 2.0 compat: replace `DataFrame.append()` with `pd.concat()` in `resources.py`

**Week 3–4: Walk-Forward Backtester**
- Build `backtest.py` with `walkforward_backtest()` function
- Integrate transaction cost model
- Compute Sharpe, max drawdown, Calmar, turnover from equity curve
- Run on 5+ years historical data; generate QuantStats tearsheet

**Week 5–6: Alpaca Paper Trading**
- Upgrade to `alpaca-py`; create `broker/alpaca_broker.py` adapter
- Implement `submit_order()`, `get_fill()`, `get_positions()`, `cancel_order()`
- Run first paper trade cycle manually

**Week 7–8: Scheduling + Basic Alerting**
- Create `run_daily.py` as main entry point
- Add NYSE holiday check, idempotency guard, failure email alert
- Set up cron/systemd on VPS

**Go/No-Go Gate**: 4 weeks paper trading; P&L within 2 std dev of backtest distribution → proceed to Phase 2.

### Phase 2: Real Money with Small Capital (2–4 weeks)
- Obtain live Alpaca keys; configure IP allowlisting; set up systemd EnvironmentFile
- Add fill-confirmation loop: submit → poll → record actual fill price in Account
- Move state files to encrypted directory with daily cloud backup
- Atomic account saves after every order batch
- Automate daily email report; replace remaining `print()` with logging

### Phase 3: Hardening (ongoing)
- JSON structured logging with log rotation
- Slack alerting integrated into logging framework
- Portfolio drift monitoring
- systemd job-level retry (`Restart=on-failure`)
- Prometheus metrics + Grafana dashboard
- Monthly key rotation calendar reminder
- Evaluate migration to GCP Cloud Run Jobs for better scheduling reliability

---

## 8. Code Changes Required

### `requirements.txt`
- Remove `alpaca_trade_api==1.2.3` (deprecated)
- Add `alpaca-py>=0.18.0`
- Add `pandas-market-calendars>=4.3.0`
- Add `python-dotenv>=1.0.0`
- Add `quantstats>=0.0.62`
- Upgrade `yfinance` to `>=0.2.40` (0.1.63 is years old, many endpoints changed)
- Upgrade `pandas` to `>=2.0` (1.1.2 from 2020; `DataFrame.append()` removed in 2.0)
- Upgrade `ortools` to `>=9.0` (8.0 from 2020)
- Remove `gurobipy==9.1.2` (no Gurobi license; not referenced in any source file)
- Pin Python `>=3.10` (3.8 EOL October 2024)

### `source/resources.py`
- Replace all `DataFrame.append()` calls with `pd.concat()` — **blocking bug on modern pandas**
- Add `TradeRecord` dataclass: fill price, fill time, broker order ID
- Modify `Account.update_account()` to accept `TradeRecord` (actual fill prices) not `Order` (optimizer prices)
- Add `Account.has_traded_today(date)` for idempotency
- Add `save_account_atomic()`: write to temp file → rename

### `source/database_handler.py`
- Fix `get_returns()` caching: key cache on `(start_date, end_date, tuple(stocks))`
- Add `get_returns_as_of(date, lookback_days)` for walk-forward backtesting
- Replace weekday date filtering with `pandas_market_calendars` NYSE schedule
- Replace `print()` with structured logging

### `source/account_manager.py`
- Remove all `plt.show()` calls (block headless execution)
- Add `save_figure=None` param to benchmark functions
- Extract `compute_risk_metrics(equity_curve, benchmark_curve, risk_free_rate)` as standalone function
- Replace `print()` with logging

### `source/opt_tools.py`
- Add solver timeout: `solver.set_time_limit(30_000)` (30s; currently unlimited — MIP can run forever)
- Add solver status check: raise exception on `INFEASIBLE`/`ABNORMAL` instead of returning garbage values
- Add solver status to returned `stats` dict
- Replace `print()` with DEBUG-level logging

### New files to create
- `source/broker/alpaca_broker.py` — Alpaca broker adapter + PaperBroker mock
- `source/backtest.py` — walk-forward backtesting loop + risk metrics
- `run_daily.py` — main scheduled entry point (market check → data update → optimize → submit → fill → save → report)

---

## Verification

1. **Backtesting**: Run `walkforward_backtest()` on 2019–2024 data; confirm Sharpe > 0 and drawdown within acceptable bounds; compare equity curve to SPY benchmark
2. **Paper trading**: Let `run_daily.py` execute for 4+ weeks against Alpaca paper environment; verify orders appear in Alpaca dashboard; verify `Account` state matches Alpaca positions
3. **Idempotency**: Manually run `run_daily.py` twice on the same day; confirm second run exits cleanly without submitting duplicate orders
4. **Failure alerting**: Kill the process mid-run; confirm email/Slack alert fires within 5 minutes
5. **State persistence**: Simulate a crash mid-write; confirm `save_account_atomic()` leaves the account file uncorrupted
6. **Live trading**: Place first real trade with $1,000–5,000; verify fill price recorded in Account matches Alpaca fill confirmation
