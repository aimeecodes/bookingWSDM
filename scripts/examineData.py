
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
# current do not have the test_data set 

# import train_data set into pandas frame
train_data = pd.read_csv(train_data_path)

# checked for NAs, none
# for col in train_data.columns:
#	print(sum(train_data[col].isna()))

start = time.process_time()

# make sure checkin and check out as cast as datetimes
train_data['checkin'] = pd.to_datetime(train_data['checkin'])
train_data['checkout'] = pd.to_datetime(train_data['checkout'])

# createvariable for each leg of trip
train_data['lengthofleg'] = (train_data['checkout'] - train_data['checkin']).dt.days

# series that holds only total length of trip data for each utrip_id
lengthofstay = train_data.groupby('utrip_id')['lengthofleg'].sum()

# series that holds a list of cities visited on each trip
# citiespertrip = train_data.groupby('utrip_id')['city_id'].nunique()
citiesvisited = train_data.groupby('utrip_id')['city_id'].agg(lambda x: list(set(x)))

# series that holds only total number of cities stopped in for each utrip_id
citiespertrip = citiesvisited.apply(lambda x: len(x))

# pull out highest number of cities visited
mostcitiesvisited = max(citiespertrip)

# series that holds what countries were visited per trip
countriesvisited = train_data.groupby('utrip_id')['hotel_country'].agg(lambda x: list(set(x)))

# series that holds total number of countries stopped in for each utrip_id
# countriespertrip = train_data.groupby('utrip_id')['hotel_country'].nunique()
countriespertrip = countriesvisited.apply(lambda x: len(x))

# pull out highest number of countries visited
mostcountriesvisited = max(countriespertrip)

# create flag that tells us if the trip crossed any country borders
internationaltrip = countriespertrip > 1

# create rearranged train_data for easier visualization
train_data_rearranged = pd.DataFrame(train_data.groupby('utrip_id')['user_id'].apply(lambda x: list(set(x))[0]))
train_data_rearranged['citiesvisited'] = citiesvisited
train_data_rearranged['citiespertrip'] = citiespertrip
train_data_rearranged['countriesvisited'] = countriesvisited
train_data_rearranged['countriespertrip'] = countriespertrip
train_data_rearranged['internationaltrip'] = internationaltrip
train_data_rearranged['lengthofstay'] = lengthofstay

# create table that holds hotel_country as key and contains all city_ids of cities inside hotel_country
citiesgroupedbycountry = train_data.groupby('hotel_country')['city_id'].agg(lambda x: list(set(x)))

# create table that holds hotel_country and contains all city_ids of cities inside hotel_country, ranked by number of trips there
citiesrankedbycountry = train_data.groupby('hotel_country')['city_id'].value_counts()

# create table that holds number of visits to hotel_country
numberofvisitstocountry = train_data['hotel_country'].value_counts()

end = time.process_time()


print('Time:', round((end - start) * 1000, 3), 'ms')