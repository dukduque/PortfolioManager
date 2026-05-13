"""
Various scripts to view and manage an account.
"""

import logging
import math
import database_handler as dbh

import datetime as dt

log = logging.getLogger(__name__)
from database_handler import DataManager
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from resources import (
    Portfolio,
    account_from_broker,
    generate_orders,
    build_account_history,
)
from opt_tools import cvar_model_ortools, default_cvar_parameters
from backtest import build_equity_curve, build_benchmark_curve


def benchmark(broker, data_manager, start_date, benchmark_symbol="SPY", save_figure=None):
    """Plot account equity curve vs a benchmark.

    Reconstructs the account from broker history, then plots total portfolio
    value against a benchmark that mirrors the same cash flows.

    Args:
        broker:           A BrokerHistory implementation.
        data_manager:     Data provider with get_prices().
        start_date:       Earliest date to pull history from.
        benchmark_symbol: Ticker for the passive benchmark (default SPY).
        save_figure:      File path to save the plot, or None to display it.
    """
    account = account_from_broker(broker, "portfolio", start_date)

    account_curve = build_equity_curve(account, data_manager)
    bench_curve = build_benchmark_curve(account, benchmark_symbol, data_manager)

    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.plot(account_curve.index, account_curve.values, color="blue",
              label="Portfolio")
    axes.plot(bench_curve.index, bench_curve.values, color="red",
              label=benchmark_symbol)
    axes.legend()

    if save_figure:
        fig.savefig(save_figure)
    else:
        plt.show()
    return fig


def piechart(broker, data_manager):
    """Plot current portfolio allocation as a pie chart.

    Args:
        broker:       A BrokerBase implementation to fetch live positions from.
        data_manager: Data provider with get_prices().
    """
    portfolio = broker.get_positions()
    assets = list(portfolio.assets)
    price = data_manager.get_prices(assets).iloc[
        -1
    ]  # Last row is the current price
    if price.isnull().sum() > 0:
        price = data_manager.get_prices(assets).iloc[-2]
    assets.sort(
        key=lambda x: portfolio.get_position(x) * price[x], reverse=True
    )
    position = np.array([portfolio.get_position(a) * price[a] for a in assets])
    portfolio_value = position.sum()
    position /= portfolio_value

    fig, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.pie(
        position,
        labels=assets,
        autopct="%1.1f%%",
        pctdistance=0.9,
        labeldistance=1.05,
        startangle=45,
    )
    axes.axis("equal")
    plt.show()

    # Pie by sectors
    sectors = defaultdict(float)
    for a in assets:
        total_position = portfolio.get_position(a) * price[a]
        marginal_position = total_position / portfolio_value
        sector = data_manager.get_metadata(a)["sector"]
        log.info("%-5s %-25s  %7.2f %5.3f%%", a, sector, total_position, marginal_position * 100)
        sectors[sector] += marginal_position
    log.info("Portfolio value: %10.2f", portfolio_value)

    fig, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.pie(
        sectors.values(),
        labels=sectors.keys(),
        autopct="%1.1f%%",
        pctdistance=0.9,
        labeldistance=1.05,
        startangle=45,
    )
    axes.axis("equal")
    plt.show()


def rebalance_porfolio(
    portfolio,
    additional_cash,
    start_date,
    end_date,
    data_file_name="close.pkl",
    metadata_file_name="metadata.pkl",
    data_manager=None,
    print_portfolio=True,
    **kwargs,
):
    """
    Return the orders to be executed to rebalance the current portfolio in
    the `account`.

    data_manager: optional pre-constructed DataManager (or compatible object).
    If None, one is created from data_file_name / metadata_file_name.
    """
    base_portfolio = portfolio

    # Data preparation specific to `base_portfolio`. In particular
    # if an asset in the portfolio is no longer listed, it is considered
    # to have a price time series of zero.
    if data_manager is None:
        data_manager = DataManager(data_file_name, metadata_file_name)
    returns = data_manager.get_returns(start_date, end_date)
    current_price = data_manager.get_prices(returns.columns).loc[end_date]
    for asset in base_portfolio.assets:
        if asset not in current_price:
            current_price[asset] = 0.0
        if asset not in returns.columns:
            returns[asset] = np.zeros(len(returns))
    returns_array = np.array(returns)

    # Read optimization params or set defaults.
    cvar_params = default_cvar_parameters()
    if "cvar_alpha" in kwargs:
        cvar_params.alpha = kwargs["cvar_alpha"]
    if "cvar_beta" in kwargs:
        cvar_params.beta = kwargs["cvar_beta"]
    fractional_stocks = False
    if "fractional_stocks" in kwargs:
        fractional_stocks = kwargs["fractional_stocks"]
    ignored_securities = []
    if "ignored_securities" in kwargs:
        ignored_securities = kwargs["ignored_securities"]
    max_weight = float(kwargs.get("max_weight", 1.0))
    portfolio_delta = float(kwargs.get("portfolio_delta", 1e9))

    opt_model = cvar_model_ortools(
        cvar_params,
        returns_array,
        current_price,
        current_portfolio=base_portfolio,
        budget=additional_cash,
        fractional=fractional_stocks,
        portfolio_delta=portfolio_delta,
        ignore=ignored_securities,
        max_weight=max_weight,
    )

    cvar_sol1, cvar_stats1 = opt_model.change_cvar_params(
        cvar_beta=cvar_params.beta)
    new_portfolio = cvar_sol1[cvar_sol1.qty > 0]
    orders = generate_orders(
        base_portfolio,
        Portfolio.create_from_vectors(new_portfolio.index, new_portfolio.qty),
        current_price,
    )
    if print_portfolio:
        for o in orders:
            log.info("  %s", o)

    # Always log the resulting portfolio table and CVaR stats.
    p = new_portfolio.copy()
    p["name"] = [
        data_manager.get_metadata(s)["name"] for s in new_portfolio.index
    ]
    p["sector"] = [
        data_manager.get_metadata(s)["sector"] for s in new_portfolio.index
    ]
    p["subsector"] = [
        data_manager.get_metadata(s)["subsector"]
        for s in new_portfolio.index
    ]
    p["value"] = p["position"].map("${:,.0f}".format)
    p["alloc"] = p["allocation"].map("{:.1%}".format)
    log.info(
        "New portfolio:\n%s",
        p[["value", "alloc", "sector", "name"]].to_string(),
    )
    log.info("CVaR stats: %s", cvar_stats1)
    return orders
