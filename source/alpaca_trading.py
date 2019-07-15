#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 12:53:19 2019

@author: dduque
"""

from os import environ

environ['APCA_API_KEY_ID'] = 'PKVUICJDJKBJKSXJX6DS'
environ['APCA_API_SECRET_KEY'] = 'xxskdLa9c8jYpUrZAf27uZUc3TNQ2UiV819zkLIa'
environ['APCA_API_BASE_URL'] = 'https://paper-api.alpaca.markets/'
import alpaca_trade_api as  tradeapi

api = tradeapi.REST('PKX51YFNEGD687QL542B', '8BCNvLMgWbjafSMi2/dz8fpNh2O2l4iYjJkMm0k6')


#orders = api.list_orders()
#print()
#for o in orders:
#    print(o.symbol, o.status)
#    #api.cancel_order(order_id=o.id)
    

# Submit a market order to buy 1 share of Apple at market price
import pickle
portfolios = pickle.load(open('./portfolios.p', 'rb'))
portfolio = portfolios[2]
for s in portfolio.index:
    print(s, portfolio.loc[s,'stock'])
    api.submit_order(
        symbol=s,
        qty=portfolio.loc[s,'stock'],
        side='buy',
        type='market',
        time_in_force='gtc'
    )
print("loaded")
#

#api.submit_order(
#    symbol='GOOGL',
#    qty=5,
#    side='buy',
#    type='market',
#    time_in_force='gtc'
#)
#
## Submit a limit order to attempt to sell 1 share of AMD at a
## particular price ($20.50) when the market opens
#api.submit_order(
#    symbol='AMD',
#    qty=1,
#    side='buy',
#    type='limit',
#    time_in_force='opg',
#    limit_price=20.50
#)