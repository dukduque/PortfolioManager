#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:36:26 2019

@author: dduque
"""


from source.opt_tools import markovitz_dro_wasserstein
import source.database_handler as dbh
import pandas as pd

year = 2005
db, data = dbh.get_returns(year)

out_dro_model = markovitz_dro_wasserstein(data, 0.001,1.001)

solution = pd.DataFrame({'allocation':out_dro_model[0]},index=db.columns)