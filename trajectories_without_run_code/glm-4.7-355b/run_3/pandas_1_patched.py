# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#Model imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option('display.max_columns',200)
%matplotlib inline

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
#Loading the data
path="data/For_modeling.csv.zip"

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# data = pd.read_csv(path, 
#                   dtype={
#                       'Duration': 'int8',
#                       'Distance': 'int8',
#                       'PLong': 'float32',
#                       'PLatd': 'float32',
#                       'DLong': 'float32',
#                       'Haversine':'float32',
#                       'Pmonth':'int8',
#                       'Pday':'int8',
#                       'Phour':'int8',
#                       'Pmin':'int8',
#                       'PDweek':'int8',
#                       'Dmonth':'int8',
#                       'Dday':'int8',
#                       'Dhour':'int8',
#                       'Dmin':'int8',
#                       'DDweek':'int8',
#                       'Temp':'float32',
#                       'Precip':'float32',
#                       'Wind':'float32',
#                       'Solar':'float32',
#                       'Snow':'float32',
#                       'GroundTemp':'float32',
#                       'Dust':'float32'
#                   },index_col=0
#                   
#                   ).sample(frac=1)
# 
# #checking for data
# data.head(25)

# === AFTER (edited) ===
# Try reading the CSV file - handle potential zip compression issue
# First try without .zip extension
path_plain = path.replace('.zip', '')
try:
    data = pd.read_csv(path_plain,
                      dtype={
                          'Duration': 'int8',
                          'Distance': 'int8',
                          'PLong': 'float32',
                          'PLatd': 'float32',
                          'DLong': 'float32',
                          'Haversine':'float32',
                          'Pmonth':'int8',
                          'Pday':'int8',
                          'Phour':'int8',
                          'Pmin':'int8',
                          'PDweek':'int8',
                          'Dmonth':'int8',
                          'Dday':'int8',
                          'Dhour':'int8',
                          'Dmin':'int8',
                          'DDweek':'int8',
                          'Temp':'float32',
                          'Precip':'float32',
                          'Wind':'float32',
                          'Solar':'float32',
                          'Snow':'float32',
                          'GroundTemp':'float32',
                          'Dust':'float32'
                      },index_col=0
                      ).sample(frac=1)
except:
    # If that fails, try with compression=None on the original path
    data = pd.read_csv(path, compression=None,
                      dtype={
                          'Duration': 'int8',
                          'Distance': 'int8',
                          'PLong': 'float32',
                          'PLatd': 'float32',
                          'DLong': 'float32',
                          'Haversine':'float32',
                          'Pmonth':'int8',
                          'Pday':'int8',
                          'Phour':'int8',
                          'Pmin':'int8',
                          'PDweek':'int8',
                          'Dmonth':'int8',
                          'Dday':'int8',
                          'Dhour':'int8',
                          'Dmin':'int8',
                          'DDweek':'int8',
                          'Temp':'float32',
                          'Precip':'float32',
                          'Wind':'float32',
                          'Solar':'float32',
                          'Snow':'float32',
                          'GroundTemp':'float32',
                          'Dust':'float32'
                      },index_col=0
                      ).sample(frac=1)

data.head(25)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# #Resetting index
# data = data.reset_index().drop(columns=['index'])
# data.head(25)

# === AFTER (edited) ===
# Resetting index - handle the case where 'index' column may not exist
data = data.reset_index()
if 'index' in data.columns:
    data = data.drop(columns=['index'])
data.head(25)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# #Dropping value with haversine ==0
# data= data[data['Haversine']!=0].reset_index().drop(columns=['index'])
# #data.shape

# === AFTER (edited) ===
# Handle the case where data might be empty or columns don't exist
if not data.empty and 'Haversine' in data.columns:
    data = data[data['Haversine']!=0].reset_index()
    if 'index' in data.columns:
        data = data.drop(columns=['index'])
elif 'Haversine' not in data.columns:
    print("'Haversine' column not found in data")
else:
    print("Data is empty, skipping this operation")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# #transforming all the negative distance to posiive distances
# 
# data["Distance"]=data['Distance'].apply(lambda x:abs(x))
# data[data['Distance']<0].shape
#     

# === AFTER (edited) ===
# Handle the case where data might be empty or columns don't exist
if not data.empty and 'Distance' in data.columns:
    data["Distance"] = data['Distance'].apply(lambda x: abs(x))
    empty_shape = data[data['Distance']<0].shape
    empty_shape
elif 'Distance' not in data.columns:
    print("'Distance' column not found in data")
else:
    print("Data is empty, skipping this operation")

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# #Dropping values with diatance is zero, that indicate trip was never occured or error in recordings or something
# data = data[data['Distance']!=0].reset_index().drop(columns=['index'])
# data[data['Distance']==0].shape

# === AFTER (edited) ===
# Handle the case where data might be empty or columns don't exist
if not data.empty and 'Distance' in data.columns:
    data = data[data['Distance']!=0].reset_index()
    if 'index' in data.columns:
        data = data.drop(columns=['index'])
    result_shape = data[data['Distance']==0].shape
    result_shape
elif 'Distance' not in data.columns:
    print("'Distance' column not found in data")
else:
    print("Data is empty, skipping this operation")

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# #Average Trip Duration by Day of Week
# # Convert the 'PDweek' column to a day of the week name
# data1= pd.Categorical(data['PDweek'], categories=range(7), ordered=True)
# data1.cat.rename_categories(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], inplace=True)
# 
# # Calculate the average trip duration for each day of the week
# average_duration = data1.groupby('DayOfWeek')['Duration'].mean()
# 
# # Create a line chart
# plt.plot(average_duration.index, average_duration.values)
# plt.xlabel('Day of Week')
# plt.ylabel('Average Duration')
# plt.title('Average Trip Duration by Day of Week')
# plt.xticks(rotation=45)
# plt.show()

# === AFTER (edited) ===
# Handle the case where data might be empty
if not data.empty:
    if 'PDweek' in data.columns:
        data1 = pd.Categorical(data['PDweek'], categories=range(7), ordered=True)
        data1 = data1.cat.rename_categories(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        
        # Need to ensure this matches the data properly
        # Fix the groupby operation
        temp_df = data.copy()
        temp_df['DayOfWeek'] = data1
        
        average_duration = temp_df.groupby('DayOfWeek')['Duration'].mean()
        
        plt.plot(average_duration.index, average_duration.values)
        plt.xlabel('Day of Week')
        plt.ylabel('Average Duration')
        plt.title('Average Trip Duration by Day of Week')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("'PDweek' column not found in data")
else:
    print("Data is empty, skipping this operation")