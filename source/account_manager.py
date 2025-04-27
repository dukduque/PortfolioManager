"""
Various scripts to view and manage an account.
"""

import math
import database_handler as dbh

import datetime as dt
from database_handler import DataManager
import pandas as pd
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from resources import (
    Portfolio,
    load_account,
    generate_orders,
    build_account_history,
)
from opt_tools import cvar_model_ortools, default_cvar_parameters


def read_account(account_name):
    account = load_account(account_name)


def benchmark2(account_name, benchmark_symbol="SPY"):
    account = load_account(account_name)
    portfolios = account.portfolios
    portfolio_dates = sorted(portfolios.keys())

    data_manager = DataManager(db_file="close.pkl")
    SPY_prices = data_manager.get_prices(benchmark_symbol)

    benchmark_quantity = 0
    account_history, history_dates, benchmark_history = [], [], []

    last_valid_price = defaultdict(float)
    last_portfolio_value = 0

    for date_ix, portfolio_date in enumerate(portfolio_dates):
        current_portfolio = portfolios[portfolio_date]
        if not current_portfolio.assets:
            continue

        assets_data = data_manager.get_prices(current_portfolio.assets)
        pd_date = pd.Timestamp(year=portfolio_date.year,
                               month=portfolio_date.month,
                               day=portfolio_date.day)

        for asset in assets_data.columns:
            price = assets_data[asset].loc[pd_date]
            if not math.isnan(price):
                last_valid_price[asset] = price

        benchmark_price_on_date = SPY_prices.loc[pd_date]
        if not math.isnan(benchmark_price_on_date):
            last_valid_price[benchmark_symbol] = benchmark_price_on_date

        portfolio_value = sum(
            current_portfolio.get_position(asset) * last_valid_price[asset]
            for asset in current_portfolio.assets
        )

        benchmark_quantity += (portfolio_value - last_portfolio_value) / \
                              last_valid_price[benchmark_symbol]
        last_portfolio_value = portfolio_value

        start_date = dt.datetime(portfolio_date.year,
                                 portfolio_date.month,
                                 portfolio_date.day)
        end_date = portfolio_dates[date_ix + 1] if date_ix + 1 < len(
            portfolio_dates) else dt.datetime.today()
        end_date = dt.datetime(end_date.year, end_date.month, end_date.day)

        assets_data = assets_data[(assets_data.index >= start_date) &
                                  (assets_data.index <= end_date)]

        for d in assets_data.index:
            total_assets_d = 0
            prices_d = assets_data.loc[d]
            for asset in prices_d.index:
                asset_price_at_d = prices_d.loc[asset]
                if not math.isnan(asset_price_at_d):
                    last_valid_price[asset] = asset_price_at_d
                total_assets_d += last_valid_price[asset] * \
                                  current_portfolio.get_position(asset)

            account_history.append(total_assets_d)

            benchmark_price_on_date = SPY_prices.loc[d]
            if not math.isnan(benchmark_price_on_date):
                last_valid_price[benchmark_symbol] = benchmark_price_on_date

            benchmark_history.append(last_valid_price[benchmark_symbol] *
                                     benchmark_quantity)
            history_dates.append(d)

    net_transactions = {}
    balance = 0
    for op_date, op_value in account.operations_history():
        balance += op_value
        pandas_date = pd.Timestamp(year=op_date.year, month=op_date.month,
                                   day=op_date.day)
        net_transactions[pandas_date] = balance

    transaction_dates = sorted(net_transactions.keys())
    cummulative_transactions = [0] * len(history_dates)

    for d_ix, d in enumerate(history_dates):
        for d_transaction in transaction_dates:
            if d > d_transaction:
                cummulative_transactions[d_ix] = net_transactions[
                    d_transaction]

    today = dt.datetime.today()
    cummulative_transactions = [net_transactions[d] for d in transaction_dates]
    transaction_dates.append(pd.Timestamp(year=today.year, month=today.month,
                                          day=today.day))
    cummulative_transactions.append(cummulative_transactions[-1])

    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.step(transaction_dates, cummulative_transactions, where="post",
              color="lightblue", alpha=0.7)
    axes.plot(history_dates, account_history, color="blue")
    axes.plot(history_dates, benchmark_history, color="red")

    plt.show()


def benchmark(account_name, benchmark_symbol="SPY"):
    """
    Benchmarks an account against `benchmark_symbol`.
    """
    account = load_account(account_name)

    file_name = "close.pkl"
    data_manager = DataManager(db_file=file_name)
    sp500_stocks = dbh.get_sp500_tickers()

    sp500_portfolios = {}
    ini_portfolio = Portfolio.create_empty()
    SPY_prices = data_manager.get_prices(benchmark_symbol)
    net_transactions = {}
    balance = 0
    for op_date, op_value in account.operations_history():
        balance += op_value
        pandas_date = pd.Timestamp(
            year=op_date.year, month=op_date.month, day=op_date.day
        )
        date_ix = np.where(SPY_prices.index == pandas_date)[0][0]
        price_on_date = SPY_prices.iloc[date_ix][0]
        qty_delta = op_value / price_on_date
        current_qty = ini_portfolio.get_position(benchmark_symbol)
        portfolio_on_date = Portfolio.create_empty()
        portfolio_on_date.modify_position(
            benchmark_symbol, current_qty + qty_delta
        )
        sp500_portfolios[op_date] = portfolio_on_date
        ini_portfolio = sp500_portfolios[op_date]
        net_transactions[pandas_date] = balance

    history_dates, sp_500_history_values = build_account_history(
        sp500_portfolios, data_manager
    )
    history_dates, history_values = build_account_history(
        account.portfolios, data_manager
    )
    sp_500_history_values.insert(0, 0)

    transaction_dates = list(net_transactions.keys())
    transaction_dates.sort()
    cummulative_transactions = [0] * len(history_dates)
    last_date_ix = 0
    for d_ix, d in enumerate(history_dates):
        for d_transaction in transaction_dates:
            if d > d_transaction:
                cummulative_transactions[d_ix] = net_transactions[d_transaction]
    today = dt.datetime.today()
    cummulative_transactions = [net_transactions[d] for d in transaction_dates]
    transaction_dates.append(
        pd.Timestamp(year=today.year, month=today.month, day=today.day)
    )
    cummulative_transactions.append(cummulative_transactions[-1])

    fig, axes = plt.subplots(ncols=1, figsize=(12, 4))
    axes.step(
        transaction_dates,
        cummulative_transactions,
        where="post",
        color="lightblue",
        alpha=0.7,
    )
    axes.plot(history_dates, history_values, color="blue")
    axes.plot(history_dates, sp_500_history_values, color="red")

    plt.show()


def piechart(account_name):
    file_name = "close.pkl"
    data_manager = DataManager(db_file=file_name)
    account = load_account(account_name)
    portfolio = account.portfolio
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
        print(
            f"{a:5s} {sector:25s}  {total_position:7.2f} {marginal_position * 100:5.3f}%"
        )
        sectors[sector] += marginal_position
    print(f"Porfolio value: {portfolio_value:10.2f}")

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
    print_portfolio=True,
    **kwargs,
):
    """
    Return the orders to be executed to rebalance the current portfolio in
    the `account`.
    """
    base_portfolio = portfolio

    # Data preparation specific to `base_portfolio`. In particular
    # if an asset in the portfolio is no longer listed, it is considered
    # to have a price time series of zero.
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

    # Create model with default parameters

    opt_model = cvar_model_ortools(
        cvar_params,
        returns_array,
        current_price,
        current_portfolio=base_portfolio,
        budget=additional_cash,
        fractional=fractional_stocks,
        portfolio_delta=0,
        ignore=ignored_securities,
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
            print(o)
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
        print(p)
        print(cvar_stats1)
    return orders
