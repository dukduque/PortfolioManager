from tests import import_source_modules
import_source_modules()

import datetime as dt
import math
import pytest
import numpy as np
import pandas as pd

from resources import (
    Account,
    Fill,
    Order,
    Portfolio,
    OPERATION_BUY,
    OPERATION_SELL,
)
from backtest import (
    BacktestConfig,
    BacktestResult,
    _cash_at_date,
    build_benchmark_curve,
    build_equity_curve,
    compute_metrics,
    run_backtest,
)
from broker.paper import CostAwarePaperBroker
from opt_tools import default_cvar_parameters
from fake import FakeDataManager, make_prices


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AA", "BB", "CC"]
SPY = "SPY"

_PRICE_DF = make_prices(TICKERS + [SPY], n_days=80)


def _fake_dm():
    return FakeDataManager(_PRICE_DF)


def _account_with_one_buy(deposit=10_000.0, qty=50, price=100.0):
    """Account: deposit on day 0, buy 50 shares of AA on day 1."""
    d0 = dt.datetime(2023, 1, 2)
    d1 = dt.datetime(2023, 1, 3)
    account = Account("test", d0)
    account.deposit(d0, deposit)
    account.update_account_from_fills(
        d1,
        [Fill("AA", qty, price, OPERATION_BUY, d1, "b-001")],
    )
    return account, d0, d1


# ---------------------------------------------------------------------------
# _cash_at_date
# ---------------------------------------------------------------------------

def test_cash_at_date_after_deposit():
    account, d0, _ = _account_with_one_buy()
    assert _cash_at_date(account, d0) == pytest.approx(10_000.0)


def test_cash_at_date_after_buy():
    account, _, d1 = _account_with_one_buy(deposit=10_000.0, qty=50, price=100.0)
    # 10_000 deposit - 50*100 buy = 5_000
    assert _cash_at_date(account, d1) == pytest.approx(5_000.0)


def test_cash_at_date_after_sell():
    account, d0, d1 = _account_with_one_buy(deposit=10_000.0, qty=50, price=100.0)
    d2 = dt.datetime(2023, 1, 4)
    account.update_account_from_fills(
        d2,
        [Fill("AA", 20, 110.0, OPERATION_SELL, d2, "b-002")],
    )
    # 5_000 + 20*110 = 7_200
    assert _cash_at_date(account, d2) == pytest.approx(7_200.0)


# ---------------------------------------------------------------------------
# build_equity_curve
# ---------------------------------------------------------------------------

def test_equity_curve_includes_cash():
    """Total value must include uninvested cash, not just equity."""
    account, d0, d1 = _account_with_one_buy(deposit=10_000.0, qty=50, price=100.0)
    dm = _fake_dm()
    curve = build_equity_curve(account, dm)

    # After the buy: equity = 50 * price_of_AA_on_d1, cash = 5_000
    # Total must be > 5_000 (equity-only would miss the cash)
    assert not curve.empty
    first_val = curve.iloc[0]
    assert first_val > 5_000.0


def test_equity_curve_is_series_with_datetime_index():
    account, _, _ = _account_with_one_buy()
    curve = build_equity_curve(account, _fake_dm())
    assert isinstance(curve, pd.Series)
    assert isinstance(curve.index[0], pd.Timestamp)


def test_equity_curve_empty_for_no_trades():
    """An account with only a deposit and no trades has no equity curve."""
    d0 = dt.datetime(2023, 1, 2)
    account = Account("test", d0)
    account.deposit(d0, 5_000.0)
    curve = build_equity_curve(account, _fake_dm())
    assert curve.empty


def test_equity_curve_grows_with_price():
    """When the held stock rises, the equity curve value must rise."""
    # Build a price DataFrame where AA goes from 100 to 200 over two days
    dates = pd.bdate_range("2023-01-02", periods=3)
    prices = pd.DataFrame(
        {"AA": [100.0, 150.0, 200.0], SPY: [400.0, 410.0, 420.0]},
        index=dates,
    )
    dm = FakeDataManager(prices)

    d0, d1 = dates[0].to_pydatetime(), dates[1].to_pydatetime()
    account = Account("test", d0)
    account.deposit(d0, 10_000.0)
    account.update_account_from_fills(
        d1, [Fill("AA", 50, 150.0, OPERATION_BUY, d1, "x")]
    )

    curve = build_equity_curve(account, dm)
    assert curve.iloc[-1] > curve.iloc[0]


def test_equity_curve_accounts_for_two_rebalances():
    """Curve is continuous across multiple transaction dates."""
    account, d0, d1 = _account_with_one_buy(deposit=10_000.0, qty=50, price=100.0)
    d2 = dt.datetime(2023, 1, 5)
    account.update_account_from_fills(
        d2,
        [Fill("BB", 10, 120.0, OPERATION_BUY, d2, "b-003")],
    )
    curve = build_equity_curve(account, _fake_dm())
    assert len(curve) > 1


# ---------------------------------------------------------------------------
# build_benchmark_curve
# ---------------------------------------------------------------------------

def test_benchmark_curve_mirrors_deposit():
    """Benchmark buys SPY with the initial deposit; curve reflects SPY price."""
    d0 = dt.datetime(2023, 1, 2)
    account = Account("test", d0)
    account.deposit(d0, 10_000.0)

    dates = pd.bdate_range("2023-01-02", periods=5)
    spy_start = 400.0
    prices = pd.DataFrame(
        {"AA": [100.0] * 5, SPY: [spy_start + i * 4 for i in range(5)]},
        index=dates,
    )
    dm = FakeDataManager(prices)
    bench = build_benchmark_curve(account, SPY, dm)

    # Expected SPY qty = 10_000 / 400.0 = 25 shares
    expected_qty = 10_000.0 / spy_start
    # Benchmark value on last day = 25 * (400 + 4*4) = 25 * 416
    assert not bench.empty
    assert bench.iloc[-1] == pytest.approx(expected_qty * prices[SPY].iloc[-1], rel=1e-4)


def test_benchmark_curve_two_deposits():
    """Two deposits each buy more SPY; total qty must be additive."""
    dates = pd.bdate_range("2023-01-02", periods=10)
    spy_prices = [400.0 + i for i in range(10)]
    prices = pd.DataFrame({"SPY": spy_prices}, index=dates)
    dm = FakeDataManager(prices)

    d0 = dates[0].to_pydatetime()
    d5 = dates[4].to_pydatetime()
    account = Account("test", d0)
    account.deposit(d0, 4_000.0)   # 4000/400 = 10 SPY
    account.deposit(d5, 2_050.0)   # 2050/404 ≈ 5.07… SPY

    bench = build_benchmark_curve(account, "SPY", dm)
    # On day 10, total qty ≈ 10 + 2050/404; value ≈ that * price[-1]
    expected_qty = 4_000.0 / 400.0 + 2_050.0 / float(prices["SPY"].iloc[4])
    assert bench.iloc[-1] == pytest.approx(expected_qty * float(prices["SPY"].iloc[-1]), rel=1e-3)


def test_benchmark_curve_withdrawal_reduces_spy():
    """A withdrawal sells SPY; benchmark value must be lower than deposit-only."""
    dates = pd.bdate_range("2023-01-02", periods=6)
    prices = pd.DataFrame(
        {"AA": [100.0] * 6, SPY: [400.0] * 6}, index=dates
    )
    dm = FakeDataManager(prices)

    d0 = dates[0].to_pydatetime()
    d3 = dates[2].to_pydatetime()
    account_full = Account("full", d0)
    account_full.deposit(d0, 8_000.0)

    account_partial = Account("partial", d0)
    account_partial.deposit(d0, 8_000.0)
    account_partial.withdraw(d3, 2_000.0)

    bench_full = build_benchmark_curve(account_full, "SPY", dm)
    bench_partial = build_benchmark_curve(account_partial, "SPY", dm)

    assert bench_partial.iloc[-1] < bench_full.iloc[-1]


def test_benchmark_curve_raises_for_no_cash_flows():
    """build_benchmark_curve raises ValueError when the account has no cash flow history."""
    d0 = dt.datetime(2023, 1, 2)
    account = Account("test", d0)
    with pytest.raises(ValueError, match="cash flow"):
        build_benchmark_curve(account, SPY, _fake_dm())


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

def _flat_curve(value=100.0, n=100):
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series([value] * n, index=dates)


def _linear_curve(start=100.0, end=120.0, n=252):
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(np.linspace(start, end, n), index=dates)


def test_metrics_total_return():
    curve = _linear_curve(100.0, 110.0)
    m = compute_metrics(curve)
    assert m["total_return"] == pytest.approx(0.10, rel=1e-3)


def test_metrics_zero_drawdown_on_monotone_curve():
    curve = _linear_curve(100.0, 120.0)
    m = compute_metrics(curve)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_metrics_drawdown_detected():
    dates = pd.bdate_range("2023-01-02", periods=6)
    # Goes up then drops: 100 → 120 → 80
    values = [100, 105, 110, 120, 100, 80]
    curve = pd.Series(values, index=dates, dtype=float)
    m = compute_metrics(curve)
    # Max drawdown from 120 to 80 = -40/120 ≈ -0.333
    assert m["max_drawdown"] == pytest.approx(-40.0 / 120.0, rel=1e-3)


def test_metrics_sharpe_positive_for_rising_curve():
    curve = _linear_curve(100.0, 130.0, n=252)
    m = compute_metrics(curve)
    assert m["sharpe"] > 0


def test_metrics_information_ratio_present_with_benchmark():
    portfolio_curve = _linear_curve(100.0, 130.0)
    benchmark_curve = _linear_curve(100.0, 115.0)
    m = compute_metrics(portfolio_curve, benchmark_curve)
    assert "information_ratio" in m


def test_metrics_returns_empty_for_short_curve():
    curve = pd.Series([100.0], index=pd.bdate_range("2023-01-02", periods=1))
    assert compute_metrics(curve) == {}


# ---------------------------------------------------------------------------
# CostAwarePaperBroker
# ---------------------------------------------------------------------------

def test_cost_aware_buy_increases_fill_price():
    broker = CostAwarePaperBroker(cost_bps=10.0)
    oid = broker.submit_order(Order("AA", 10, 100.0, OPERATION_BUY))
    fill = broker.get_fill(oid)
    assert fill.fill_price == pytest.approx(100.0 * (1 + 10 / 10_000))


def test_cost_aware_sell_decreases_fill_price():
    initial = Portfolio.create_from_vectors(["AA"], [20])
    broker = CostAwarePaperBroker(cost_bps=10.0, initial_portfolio=initial)
    oid = broker.submit_order(Order("AA", 10, 100.0, OPERATION_SELL))
    fill = broker.get_fill(oid)
    assert fill.fill_price == pytest.approx(100.0 * (1 - 10 / 10_000))


def test_cost_aware_zero_bps_same_as_paper():
    broker = CostAwarePaperBroker(cost_bps=0.0)
    oid = broker.submit_order(Order("AA", 5, 200.0, OPERATION_BUY))
    fill = broker.get_fill(oid)
    assert fill.fill_price == pytest.approx(200.0)


def test_cost_aware_position_qty_unchanged():
    """Transaction cost changes price, not quantity."""
    broker = CostAwarePaperBroker(cost_bps=50.0)
    broker.submit_order(Order("AA", 10, 100.0, OPERATION_BUY))
    assert broker.get_positions().get_position("AA") == 10


# ---------------------------------------------------------------------------
# run_backtest (end-to-end on synthetic data)
# ---------------------------------------------------------------------------

def test_run_backtest_returns_result():
    prices = make_prices(TICKERS + [SPY], n_days=80)
    dm = FakeDataManager(prices)
    schedule = [prices.index[30].to_pydatetime(), prices.index[60].to_pydatetime()]
    config = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    result = run_backtest(config, dm, benchmark_symbol=SPY)
    assert isinstance(result, BacktestResult)


def test_run_backtest_equity_curve_nonempty():
    prices = make_prices(TICKERS + [SPY], n_days=80)
    dm = FakeDataManager(prices)
    schedule = [prices.index[30].to_pydatetime(), prices.index[60].to_pydatetime()]
    config = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    result = run_backtest(config, dm, benchmark_symbol=SPY)
    assert not result.equity_curve.empty


def test_run_backtest_benchmark_curve_nonempty():
    prices = make_prices(TICKERS + [SPY], n_days=80)
    dm = FakeDataManager(prices)
    schedule = [prices.index[30].to_pydatetime(), prices.index[60].to_pydatetime()]
    config = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    result = run_backtest(config, dm, benchmark_symbol=SPY)
    assert not result.benchmark_curve.empty


def test_run_backtest_trades_recorded():
    prices = make_prices(TICKERS + [SPY], n_days=80)
    dm = FakeDataManager(prices)
    schedule = [prices.index[30].to_pydatetime(), prices.index[60].to_pydatetime()]
    config = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    result = run_backtest(config, dm, benchmark_symbol=SPY)
    assert not result.trades.empty


def test_run_backtest_metrics_populated():
    prices = make_prices(TICKERS + [SPY], n_days=80)
    dm = FakeDataManager(prices)
    schedule = [prices.index[30].to_pydatetime(), prices.index[60].to_pydatetime()]
    config = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    result = run_backtest(config, dm, benchmark_symbol=SPY)
    assert "sharpe" in result.metrics
    assert "max_drawdown" in result.metrics
    assert "total_return" in result.metrics


def test_run_backtest_with_transaction_costs_reduces_cash():
    """Higher transaction costs should leave less cash after the same trades."""
    prices = make_prices(TICKERS + [SPY], n_days=80)
    dm = FakeDataManager(prices)
    schedule = [prices.index[30].to_pydatetime()]

    config_free = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        transaction_cost_bps=0.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    config_costly = BacktestConfig(
        rebalance_schedule=schedule,
        lookback_days=25,
        initial_cash=10_000.0,
        transaction_cost_bps=100.0,
        cvar_params=default_cvar_parameters(),
        fractional_shares=True,
    )
    result_free = run_backtest(config_free, dm, benchmark_symbol=SPY)
    result_costly = run_backtest(config_costly, dm, benchmark_symbol=SPY)

    # The costly run must have less remaining cash than the free run
    assert result_costly.account.cash_onhand < result_free.account.cash_onhand
