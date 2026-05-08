"""Walk-forward backtesting and portfolio performance analysis.

Public API:
    BacktestConfig       — configuration for run_backtest
    BacktestResult       — output of run_backtest
    build_equity_curve   — daily total value (equity + cash) from an Account
    build_benchmark_curve— daily benchmark value matching account cash flows
    compute_metrics      — Sharpe, Sortino, max drawdown, Calmar, info ratio
    run_backtest         — schedule-driven walk-forward simulation
"""
import datetime as dt
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from opt_tools import CvarParameters, cvar_model_ortools, default_cvar_parameters
from resources import (
    OPERATION_BUY,
    OPERATION_SELL,
    Account,
    Order,
    Portfolio,
    build_account_history,
    generate_orders,
)


# ---------------------------------------------------------------------------
# Configuration and result types
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    """Configuration for a schedule-driven walk-forward backtest."""
    rebalance_schedule: List[dt.datetime]
    lookback_days: int = 504
    initial_cash: float = 100_000.0
    transaction_cost_bps: float = 5.0
    cvar_params: Optional[CvarParameters] = None
    fractional_shares: bool = True
    universe: Optional[List[str]] = None
    # Max total shares that may be reduced across all positions per rebalance.
    # None = no restriction (full rebalancing allowed).
    portfolio_delta: Optional[float] = None


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest."""
    equity_curve: pd.Series
    benchmark_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict
    account: Account


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_ts(d) -> pd.Timestamp:
    return pd.Timestamp(d)


def _cash_at_date(account: Account, target_date) -> float:
    """Reconstruct cash balance at any historical date.

    Sums deposits/withdrawals from cash_flow and buy/sell costs from
    transactions that occurred on or before target_date.
    """
    target_ts = _to_ts(target_date)
    cash = 0.0
    for _, row in account.cash_flow.iterrows():
        if _to_ts(row["datetime"]) <= target_ts:
            cash += row["amount"] if row["type"] == "deposit" else -row["amount"]
    for _, row in account.transactions.iterrows():
        if _to_ts(row["datetime"]) <= target_ts:
            if row["operation"] == OPERATION_BUY:
                cash -= row["qty"] * row["price"]
            elif row["operation"] == OPERATION_SELL:
                cash += row["qty"] * row["price"]
    return cash


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def build_equity_curve(account: Account, data_manager) -> pd.Series:
    """Daily time series of total account value: equity + cash.

    Cash is reconstructed at each portfolio-snapshot date from the account's
    cash_flow (deposits/withdrawals) and transactions (buy/sell costs).
    Unlike the legacy build_account_history, this includes uninvested cash so
    the curve correctly reflects deposits, partial deployment, and withdrawals.

    Args:
        account: Account with portfolios, cash_flow, and transactions history.
        data_manager: object with get_prices(assets) returning a DataFrame.

    Returns:
        pd.Series indexed by date with name "portfolio_value".
    """
    portfolio_dates = sorted(account.portfolios.keys())
    all_dates: List = []
    all_values: List[float] = []
    last_valid: Dict[str, float] = defaultdict(float)

    for date_ix, tx_date in enumerate(portfolio_dates):
        portfolio = account.portfolios[tx_date]
        if not portfolio.assets:
            continue

        cash = _cash_at_date(account, tx_date)
        end_date = (
            portfolio_dates[date_ix + 1]
            if date_ix + 1 < len(portfolio_dates)
            else dt.datetime.today()
        )

        assets = list(portfolio.assets)
        prices_df = data_manager.get_prices(assets)
        if isinstance(prices_df, pd.Series):
            prices_df = prices_df.to_frame()

        start_ts, end_ts = _to_ts(tx_date), _to_ts(end_date)
        slice_ = prices_df[(prices_df.index >= start_ts) & (prices_df.index < end_ts)]

        for d in slice_.index:
            equity = 0.0
            for asset in assets:
                p = float(slice_.loc[d, asset])
                if not math.isnan(p):
                    last_valid[asset] = p
                equity += last_valid[asset] * portfolio.get_position(asset)
            all_dates.append(d)
            all_values.append(equity + cash)

    return pd.Series(all_values, index=all_dates, name="portfolio_value")


def build_benchmark_curve(
    account: Account, benchmark_symbol: str, data_manager
) -> pd.Series:
    """Daily benchmark portfolio value that mirrors the account's cash flows.

    For each deposit in account.cash_flow, the benchmark buys
    (deposit / benchmark_price) shares of benchmark_symbol.  For each
    withdrawal, it sells the equivalent amount.  Rebalances (sell A / buy B)
    involve no new external cash so the benchmark is unaffected — giving a
    fair "what if I had just bought SPY" comparison.

    Args:
        account:          Account whose cash_flow drives the benchmark.
        benchmark_symbol: Ticker to use as the benchmark (e.g. "SPY").
        data_manager:     object with get_prices(assets) returning a DataFrame.

    Returns:
        pd.Series indexed by date with name benchmark_symbol.
    """
    if account.cash_flow.empty:
        raise ValueError(
            "Cannot build benchmark curve: account has no cash flow history. "
            "The broker may not support get_cash_flows() — check "
            "broker.supports_cash_flow_history before calling this function."
        )

    bench_prices = data_manager.get_prices(benchmark_symbol)
    if isinstance(bench_prices, pd.DataFrame):
        bench_prices = bench_prices.iloc[:, 0]

    bench_qty = 0.0
    bench_portfolios: Dict = {}

    for _, row in account.cash_flow.sort_values("datetime").iterrows():
        flow_ts = _to_ts(row["datetime"])
        delta = row["amount"] if row["type"] == "deposit" else -row["amount"]

        # Use the price on or after the cash-flow date.
        future = bench_prices[bench_prices.index >= flow_ts]
        if future.empty:
            continue
        price = float(future.iloc[0])
        if math.isnan(price) or price <= 0:
            continue

        bench_qty += delta / price
        p = Portfolio.create_empty()
        if bench_qty > 0:
            p.modify_position(benchmark_symbol, bench_qty)
        bench_portfolios[row["datetime"]] = p

    if not bench_portfolios:
        return pd.Series(dtype=float, name=benchmark_symbol)

    dates_list, values_list = build_account_history(bench_portfolios, data_manager)
    return pd.Series(values_list, index=dates_list, name=benchmark_symbol)


def compute_metrics(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> Dict:
    """Compute standard risk/return metrics from a daily equity curve.

    Args:
        equity_curve:    Daily portfolio values (pd.Series indexed by date).
        benchmark_curve: Optional daily benchmark values for relative metrics.
        risk_free_rate:  Annualised risk-free rate (e.g. 0.05 = 5 %).

    Returns dict with keys: total_return, annualized_return, annualized_vol,
    sharpe, sortino, max_drawdown, calmar, and information_ratio (if benchmark
    is provided).
    """
    if len(equity_curve) < 2:
        return {}

    daily_ret = equity_curve.pct_change().dropna()
    ann = 252
    daily_rf = risk_free_rate / ann

    total_return = float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
    n_years = len(daily_ret) / ann
    ann_return = float((1 + total_return) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(daily_ret.std()) * math.sqrt(ann)

    excess = daily_ret - daily_rf
    sharpe = (
        float(excess.mean()) / float(excess.std()) * math.sqrt(ann)
        if float(excess.std()) > 0 else 0.0
    )

    downside = excess[excess < 0]
    down_std = float(downside.std()) * math.sqrt(ann) if len(downside) > 1 else 0.0
    sortino = (ann_return - risk_free_rate) / down_std if down_std > 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    metrics: Dict = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
    }

    if benchmark_curve is not None and len(benchmark_curve) >= 2:
        bench_ret = benchmark_curve.pct_change().dropna()
        common = daily_ret.index.intersection(bench_ret.index)
        if len(common) > 1:
            alpha = daily_ret[common] - bench_ret[common]
            ir = (
                float(alpha.mean()) / float(alpha.std()) * math.sqrt(ann)
                if float(alpha.std()) > 0 else 0.0
            )
            metrics["information_ratio"] = ir

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

def run_backtest(
    config: BacktestConfig,
    data_manager,
    benchmark_symbol: str = "SPY",
) -> BacktestResult:
    """Schedule-driven walk-forward backtest.

    On each date in config.rebalance_schedule the CVaR optimizer is called
    with the returns from [rebalance_date - lookback_days, rebalance_date].
    Orders are filled via CostAwarePaperBroker and the simulated Account is
    updated with actual fill prices.

    Args:
        config:           BacktestConfig describing the simulation.
        data_manager:     DataManager (or FakeDataManager) for price data.
        benchmark_symbol: Ticker used as the passive benchmark.

    Returns:
        BacktestResult with equity curve, benchmark curve, trades, metrics,
        and the full simulated Account.
    """
    from broker.paper import CostAwarePaperBroker

    cvar_params = config.cvar_params or default_cvar_parameters()
    schedule = sorted(config.rebalance_schedule)

    account = Account("backtest", schedule[0])
    account.deposit(schedule[0], config.initial_cash)
    broker = CostAwarePaperBroker(cost_bps=config.transaction_cost_bps)

    for rebalance_date in schedule:
        lookback_start = rebalance_date - dt.timedelta(days=config.lookback_days)
        universe = config.universe or []

        try:
            returns = data_manager.get_returns(lookback_start, rebalance_date, stocks=universe)
        except Exception:
            continue

        if returns is None or returns.empty or len(returns.columns) < 2:
            continue

        prices_df = data_manager.get_prices(list(returns.columns))
        if isinstance(prices_df, pd.Series):
            prices_df = prices_df.to_frame()
        rebalance_ts = _to_ts(rebalance_date)
        if rebalance_ts in prices_df.index:
            current_price = prices_df.loc[rebalance_ts]
        else:
            past = prices_df[prices_df.index <= rebalance_ts]
            if past.empty:
                continue
            current_price = past.iloc[-1]

        portfolio = account.portfolio
        for asset in list(portfolio.assets):
            if asset not in current_price.index:
                current_price[asset] = 0.0
            if asset not in returns.columns:
                returns = returns.copy()
                returns[asset] = np.zeros(len(returns))

        total_value = account.cash_onhand + sum(
            float(current_price.get(a, 0)) * portfolio.get_position(a)
            for a in portfolio.assets
        )
        portfolio_delta = (
            config.portfolio_delta
            if config.portfolio_delta is not None
            else total_value  # no sell restriction
        )

        try:
            opt_model = cvar_model_ortools(
                cvar_params,
                np.array(returns),
                current_price,
                current_portfolio=portfolio,
                budget=account.cash_onhand,
                fractional=config.fractional_shares,
                portfolio_delta=portfolio_delta,
            )
            cvar_sol, _ = opt_model.change_cvar_params(cvar_beta=cvar_params.beta)
        except Exception:
            continue

        new_df = cvar_sol[cvar_sol.qty > 0]
        if new_df.empty:
            continue

        new_port = Portfolio.create_from_vectors(
            new_df.index.tolist(), new_df.qty.tolist()
        )
        orders = generate_orders(portfolio, new_port, current_price)
        if not orders:
            continue

        fills = []
        for order in orders:
            try:
                oid = broker.submit_order(order)
                fill = broker.get_fill(oid)
                if fill:
                    fills.append(fill)
            except ValueError:
                continue

        if fills:
            account.update_account_from_fills(rebalance_date, fills)

    equity_curve = build_equity_curve(account, data_manager)
    benchmark_curve = build_benchmark_curve(account, benchmark_symbol, data_manager)
    metrics = compute_metrics(equity_curve, benchmark_curve)

    return BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        trades=account.transactions.copy(),
        metrics=metrics,
        account=account,
    )
