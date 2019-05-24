#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:32:31 2019

@author: dduque
"""

'''
Setup paths
'''
import os 
import sys
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.append(parent_path)
path_to_data = os.path.abspath(os.path.join(parent_path,  'data'))

'''
Setup libraries
'''
import pandas as pd
import pandas_datareader.robinhood as rh
import pandas_datareader.data as web
import datetime
import pickle


def load_database(DB_file_name):
    path_to_database = os.path.join(path_to_data,DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists == False:
        return None
    
    try:
        return pd.read_pickle(path_to_database)
    except Exception as e: 
        print(e)
        
def safe_database(DB_file_name, BD):
    path_to_database = os.path.join(path_to_data,DB_file_name)
    exists = os.path.isfile(path_to_database)
    if exists:
        #TODO: Make a copy of the existing file
        pass
    BD.to_pickle(path_to_database)



def create_database():
    symbols_df = web.get_nasdaq_symbols()
    symbols_df = symbols_df[symbols_df.ETF==False]
    sym_list = list(symbols_df.index)
    db = web.DataReader(sym_list, 'yahoo', start='2019-01-10')
    



    
        


F = web.DataReader('F', 'yahoo')
