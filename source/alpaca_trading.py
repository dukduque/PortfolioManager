#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:53:19 2019

@author: dduque
"""
from os import environ
from opt_tools import cvar_model_pulp
import database_handler as dbh 
import datetime as dt
import pickle
import alpaca_trade_api as tradeapi
environ['APCA_API_KEY_ID'] = 'PKN0EHU7EQ2ML1IA0KRI'
environ['APCA_API_SECRET_KEY'] = '/cmrcUS7eoD2MSWY5OZQAI6VEolDKEspevu5KIJd'
environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets/'


#api = tradeapi.REST('PKN0EHU7EQ2ML1IA0KRI','/cmrcUS7eoD2MSWY5OZQAI6VEolDKEspevu5KIJd')
api = tradeapi.REST()


# orders = api.list_orders()
# print(orders)
# print()
# for o in orders:
#    print(o.symbol, o.status)
#    #api.cancel_order(order_id=o.id)

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


def rebalance(api, additions=0, stock_univers='sp500'):
    '''
    Args:
        api (Object): trading api.
        additions (float): additions in capital.
        stock_universe (str): set of stocks to consider.
            available options are sp500 (default) and rusell1000.
    '''

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
    opt_model = cvar_model_pulp(data, price, budget=new_mkt_val, fractional=True)

if __name__ == '__main__':
    api = tradeapi.REST()
    #dbh.run_update_process()
    rebalance(api)
