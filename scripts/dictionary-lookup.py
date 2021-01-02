
###############################################################################
#                                                                             #
#     nix-shell should run this to initialize the booking data train and      #
#     test sets, as well as loading necessary imports like pandas,            #
#     matplotlib, seaborn, and scikit-learn                                   #
#                                                                             #
###############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import statsmodels.api as sm
import datetime as dt
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from numpy.polynomial.polynomial import polyfit

# read in the datasets
train_data_path = '~/Projects/BookingWSDM/booking_train_set.csv'
unique_trips_path = '~/Projects/BookingWSDM/citiesvisitedpertrip.csv'
# current do not have the test_data set 

# import train_data set into pandas frame
train_data = pd.read_csv(train_data_path)
citiesvisiteduniquetrip = pd.read_csv(unique_trips_path).set_index('utrip_id')

# checked for NAs, none
# for col in train_data.columns:
#	print(sum(train_data[col].isna()))

# start = time.process_time()

# initialize dictionary that holds names of all cities
city_connections = {}.fromkeys(train_data['city_id'].unique(), [{}.fromkeys(train_data['city_id'].unique(), 0), {}.fromkeys(train_data['city_id'].unique(), 0)])

# now for each utrip_id in citiesvisited, need a function to pass the list of trips into
# which should return pairs (outgoing and incoming)
