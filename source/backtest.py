#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:47:36 2019

@author: dduque

Implemnets backtesting function for a given porfolio
"""

import database_handler as dbh
import matplotlib.pyplot as plt
import datetime as dt


def backtest(portfolio, test_data):
    '''
    Run a single backtest
    Args:
        portfolio (DataFrame): a dataframe with a porfolio. The df should have
        the symbols as index, and the corresponding price, number of stocks, 
        position (price*stocks), and allocation (position/total investment).
        test_data (DataFrame): a dataframe of stock returns.
    '''
    stocks = portfolio[portfolio.stock>0].index
    data = test_data[stocks]
    position = portfolio[portfolio.stock>0].position

    
    cash_ini = portfolio.position.sum()
    cash_path = [cash_ini]
    for d in data.index:
        position = data.loc[d]*(position)
        cash_path.append(position.sum())
    
    cash_end = position.sum()
    
    return cash_end, cash_path

def run_backtest(portfolio, stock_prices, start, end, test_length = 1, plot=False):
    stocks = portfolio[portfolio.position>0].index
    stock_data = stock_prices[stocks]
   
    date_diff = end - start
    n_test = int(date_diff.days/(test_length*365))
    start_y = start
    cash_paths = []
    for y in range(n_test):
        start_y = start + dt.timedelta(test_length*365*y)
        end_y = start_y + dt.timedelta(test_length*365)
        print(start_y, end_y)
        stock_prices_y = stock_data[(stock_data.index>=start_y) & (stock_data.index<=end_y)]
        stock_returns_y = stock_prices_y.apply(dbh.quotien_diff, axis = 0)
        cash_end, cash_path = backtest(portfolio, stock_returns_y)
        print(cash_end)
        cash_paths.append(cash_path)
    plt.show()
    return cash_paths
        
    
def plot_backtests(portfolio_cash_paths):
    my_colors = ['r','b','k','c','g']
    fig, ax = plt.subplots(figsize=(7, 4), dpi=100)
    for (i,pcp) in enumerate(portfolio_cash_paths):
        for cp in pcp:
            ax.plot(cp, c=my_colors[i])
    plt.tight_layout()
    plt.show()
    
        

        
    


