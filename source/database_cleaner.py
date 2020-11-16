#This script has two purposes
'''
Setup libraries and import files
'''
#This next code is a hack to force VS Code to detect the fsource folder and add it to the working directory
import sys, os
path_to_file = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(path_to_file, os.pardir))
sys.path.append(parent_path)

from source import database_handler as dbh
'''
Define the data sets to be cleaned, date range, and cleaning criteria
'''
data = dbh.load_database('close_2020-09-07.pkl')
#When are we starting the analysis?
start_date = '2014-01-01'
#When are we finishing the analysis?
end_date = '2019-01-01'
#What is the minimum return that we will accept?
r_min = 0.01
#What is the maximum return that we will accept?
r_max = 3
'''
Clean database of outlier returns
'''
#Narrow the database to the desired date
data_clean = data.loc[start_date:end_date, ]
#Eliminate stocks that don't have complete data
data_clean = data_clean.dropna(axis='columns')
#Calculate returns of the remaining stocks
data_r = dbh.quotien_diff(data_clean)
#Identify the stocks that are outliers based on the previously defined criteria
data_bool = data_r < r_min
sum_rows = data_bool.sum(0)
sum_rows = sum_rows[sum_rows >= 1]
bad_stocks = set(sum_rows.index)
bad_stocks = list(bad_stocks)

data_bool = data_r > r_max
sum_rows = data_bool.sum(0)
sum_rows = sum_rows[sum_rows >= 1]
bad_stocks_max = set(sum_rows.index)
bad_stocks_max = list(bad_stocks_max)

#Create a list of all the stocks we need to remove
bad_stocks_final = bad_stocks + bad_stocks_max
#Remove duplicates
bad_stocks_final = list(dict.fromkeys(bad_stocks_final))

#Drop outlier stocks from database
data_clean = data_clean.drop(columns=bad_stocks_final)
#Calculate returns on final database
data_clean_r = dbh.quotien_diff(data_clean)

#check intersection with major indexes
sp500 = set(dbh.save_sp500_tickers())
r1000 = set(dbh.save_rusell1000_tickers())
intersec_sp500 = sp500.intersection(bad_stocks_final)
intersec_r1000 = r1000.intersection(bad_stocks_final)

hola
hola 2!