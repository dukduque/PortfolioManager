#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:53:19 2019

@author: dduque
"""
from os import environ
from opt_tools import cvar_model_pulp
import numpy as np
import database_handler as dbh 
import datetime as dt
import pickle
import time
import alpaca_trade_api as tradeapi

environ['APCA_API_KEY_ID'] = 'PKRHWE09P8OTF9X5592S'
environ['APCA_API_SECRET_KEY'] = 'dS483sqISw6z2T5Aco5WZubzW4BP8yrGgWXp7xuq'
environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets/'

#api = tradeapi.REST()

def cancel_all_orders(alpaca_api):
    '''
    All orders in alpaca that have not been
    resoluted will be canceled.
    '''
    orders = alpaca_api.list_orders()
    for o in orders:
        print(o.symbol, o.status)
        api.cancel_order(order_id=o.id)

def rebalance(api, additions=0, stock_univers='sp500'):
    '''
    Args:
        api (Object): trading api.
        additions (float): additions in capital.
        stock_universe (str): set of stocks to consider.
            available options are sp500 (default) and rusell1000.
    '''
    # Account
    alpaca_account = api.get_account()

    # Get current portfolio
    current_portfolio = api.list_positions()
    mkt_val = sum(float(position.market_value) for position in current_portfolio)
    new_mkt_val = mkt_val + additions

    today = dt.datetime.today()
    my_date = today.replace(today.year - 2)
    stocks = dbh.save_sp500_tickers() if stock_univers == 'sp500' else dbh.save_rusell1000_tickers()
    db, data = dbh.get_returns(
        data_file='close.pkl', start_date=my_date,
        stocks=stocks)
    '''
    Create modelwith default parameters
    '''
    price = db.iloc[-1]  # Last row is the current price
    opt_model = cvar_model_pulp(data, price, budget=new_mkt_val, fractional=False)
    new_portfolio, new_stats = opt_model.optimize()
    # Analyze new portfolio.
    for p in current_portfolio:
        if p.symbol in new_portfolio.index:
            new_qty = new_portfolio['qty'][p.symbol] - float(p.qty)
            new_portfolio['qty'][p.symbol] = np.abs(new_qty)
            new_portfolio['side'][p.symbol] = 'buy' if new_qty > 0 else 'sell'
        else:
            new_portfolio.loc[p.symbol] = [float(p.current_price), float(p.qty), float(p.market_value), 0, 'sell']
    new_portfolio = new_portfolio[new_portfolio.qty > 0]
    # Submit the changes.
    # Selling
    sell = new_portfolio[new_portfolio.side == 'sell']
    selling_orders = submit_orders(sell)
    success, failed = comfirm_orders(selling_orders)
    # Buying
    buy = new_portfolio[new_portfolio.side == 'buy']
    buying_orders = submit_orders(buy)
    success, faied = comfirm_orders(buying_orders)

    current_portfolio = api.list_positions()
    mkt_val = sum(float(position.market_value) for position in current_portfolio)
    for p in current_portfolio:
        print('%6s %10s %10s %10s' % (p.symbol, p.cost_basis, p.qty, p.market_value))
    print('Mkt value: ', mkt_val)

def submit_orders(orders):
    trading_orders = []
    for s in orders.index:
        print(s)
        order = s
        order = api.submit_order(
            symbol=s,
            qty=orders.loc[s, 'qty'],
            side=orders.loc[s, 'side'],
            type='market',
            time_in_force='gtc'
        )
        trading_orders.append(order)
    return trading_orders

def comfirm_orders(trading_orders):
    complete = False
    successful_orders = []
    failed_orders = []
    while not complete:
        for so in trading_orders:
            if so.status == 'filled' or so.status == 'new':
                successful_orders.append(so)
            elif so.status == 'canceled':
                failed_orders.append(so)
        complete = len(failed_orders + successful_orders) == len(trading_orders)
        print('Total filled: %i, total canceled: %i, total orders: %i' %(len(successful_orders), len(failed_orders), len(trading_orders)))
        time.sleep(10)
    return successful_orders, failed_orders

if __name__ == '__main__':
    api = tradeapi.REST()
    #dbh.run_update_process()
    rebalance(api)



# Code snippets

# Submit a market order to buy 1 share of Apple at market price

# portfolios = pickle.load(open('./portfolios.p', 'rb'))
# portfolio = portfolios[2]
# for s in portfolio.index:
#     print(s, portfolio.loc[s, 'stock'])
#     api.submit_order(
#         symbol=s,
#         qty=portfolio.loc[s, 'stock'],
#         side='buy',
#         type='market',
#         time_in_force='gtc'
#     )
# print("loaded")
#

# api.submit_order(
#    symbol='GOOGL',
#    qty=5,
#    side='buy',
#    type='market',
#    time_in_force='gtc'
# )
#
# Submit a limit order to attempt to sell 1 share of AMD at a
# particular price ($20.50) when the market opens
# api.submit_order(
#    symbol='AMD',
#    qty=1,
#    side='buy',
#    type='limit',
#    time_in_force='opg',
#    limit_price=20.50
# )
