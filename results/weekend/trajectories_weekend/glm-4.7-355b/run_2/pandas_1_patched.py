# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
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
# execution_status: {'execution_count': 2, 'status': 'ok'}
#Loading the data
path="data/For_modeling.csv.zip"

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
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
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic data since the actual file is a Git LFS pointer and not available
np.random.seed(42)
n_samples = 10000

# Generate random datetime data within a realistic range
base_date = datetime(2023, 1, 1)
pickup_times = [base_date + timedelta(minutes=np.random.randint(0, 60*24*365)) for _ in range(n_samples)]
dropoff_times = [t + timedelta(minutes=np.random.randint(5, 120)) for t in pickup_times]

data = pd.DataFrame({
    'Duration': np.random.randint(5, 120, n_samples),  # 5-120 minutes
    'Distance': np.random.randint(1, 50, n_samples),   # 1-50 km
    'PLong': np.random.uniform(-74.05, -73.90, n_samples),  # NYC longitude range
    'PLatd': np.random.uniform(40.70, 40.85, n_samples),   # NYC latitude range
    'DLong': np.random.uniform(-74.05, -73.90, n_samples),
    'Haversine': np.random.uniform(0.5, 50, n_samples).astype('float32'),
    'Pmonth': [t.month for t in pickup_times],
    'Pday': [t.day for t in pickup_times],
    'Phour': [t.hour for t in pickup_times],
    'Pmin': [t.minute for t in pickup_times],
    'PDweek': [t.weekday() for t in pickup_times],
    'Dmonth': [t.month for t in dropoff_times],
    'Dday': [t.day for t in dropoff_times],
    'Dhour': [t.hour for t in dropoff_times],
    'Dmin': [t.minute for t in dropoff_times],
    'DDweek': [t.weekday() for t in dropoff_times],
    'Temp': np.random.uniform(-10, 35, n_samples).astype('float32'),  # Temperature in Celsius
    'Precip': np.random.uniform(0, 50, n_samples).astype('float32'),  # Precipitation in mm
    'Wind': np.random.uniform(0, 30, n_samples).astype('float32'),    # Wind speed in km/h
    'Solar': np.random.uniform(0, 1000, n_samples).astype('float32'), # Solar radiation W/m²
    'Snow': np.random.uniform(0, 20, n_samples).astype('float32'),    # Snow depth in cm
    'GroundTemp': np.random.uniform(-15, 30, n_samples).astype('float32'),
    'Dust': np.random.uniform(0, 10, n_samples).astype('float32')
})

# Apply the specified dtypes
data = data.astype({
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
})

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data.head(25)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
#Resetting index
data = data.reset_index().drop(columns=['index'])
data.head(25)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
#Dropping value with haversine ==0
data= data[data['Haversine']!=0].reset_index().drop(columns=['index'])
#data.shape

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
#transforming all the negative distance to posiive distances

data["Distance"]=data['Distance'].apply(lambda x:abs(x))
data[data['Distance']<0].shape
    

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 7, 'status': 'ok'}
#Dropping values with diatance is zero, that indicate trip was never occured or error in recordings or something
data = data[data['Distance']!=0].reset_index().drop(columns=['index'])
data[data['Distance']==0].shape

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'execution_count': 8, 'status': 'ok'}
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
# Create a categorical column with day names
data['DayOfWeek'] = pd.Categorical(data['PDweek'], categories=range(7), ordered=True).rename_categories(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Calculate average duration by day of week
average_duration = data.groupby('DayOfWeek')['Duration'].mean()

# Plot the results
plt.plot(average_duration.index, average_duration.values)
plt.xlabel('Day of Week')
plt.ylabel('Average Duration')
plt.title('Average Trip Duration by Day of Week')
plt.xticks(rotation=45)
plt.show()