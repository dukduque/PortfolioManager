#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:15:56 2019

@author: dduque
"""

import sys
import os

import argparse


def dh_parse_arguments():
    parser = argparse.ArgumentParser("Datahandler parser")
    parser.add_argument("-a", choices=['u', 'd', 'sp500'], default='u', help='Action to perform')
    parser.add_argument("-db_file", type=str, default='close.pkl', help='File name of the database to update')
    parser.add_argument("-n_proc", default=4, type=int, help='Number of processor to use')
    parser.add_argument("-days_back", default=1, type=int, help='Number of days to replace in the update')
    args = parser.parse_args()
    return args
