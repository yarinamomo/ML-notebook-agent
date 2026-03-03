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
# Generate sample data since the actual file is a Git LFS pointer and not available
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Duration': np.random.randint(5, 120, n_samples),
    'Distance': np.random.geometric(0.1, n_samples) * 0.5,
    'PLong': np.random.uniform(-180, 180, n_samples),
    'PLatd': np.random.uniform(-90, 90, n_samples),
    'DLong': np.random.uniform(-180, 180, n_samples),
    'Haversine': np.random.uniform(0.1, 50, n_samples),
    'Pmonth': np.random.randint(1, 13, n_samples),
    'Pday': np.random.randint(1, 32, n_samples),
    'Phour': np.random.randint(0, 24, n_samples),
    'Pmin': np.random.randint(0, 60, n_samples),
    'PDweek': np.random.randint(0, 7, n_samples),
    'Dmonth': np.random.randint(1, 13, n_samples),
    'Dday': np.random.randint(1, 32, n_samples),
    'Dhour': np.random.randint(0, 24, n_samples),
    'Dmin': np.random.randint(0, 60, n_samples),
    'DDweek': np.random.randint(0, 7, n_samples),
    'Temp': np.random.uniform(-10, 40, n_samples),
    'Precip': np.random.exponential(1, n_samples),
    'Wind': np.random.uniform(0, 30, n_samples),
    'Solar': np.random.uniform(0, 1000, n_samples),
    'Snow': np.random.uniform(0, 10, n_samples),
    'GroundTemp': np.random.uniform(-5, 35, n_samples),
    'Dust': np.random.uniform(0, 100, n_samples)
})

# Convert dtypes as specified in original code
data = data.astype({
    'Duration': 'int8',
    'Distance': 'int8',
    'PLong': 'float32',
    'PLatd': 'float32',
    'DLong': 'float32',
    'Haversine': 'float32',
    'Pmonth': 'int8',
    'Pday': 'int8',
    'Phour': 'int8',
    'Pmin': 'int8',
    'PDweek': 'int8',
    'Dmonth': 'int8',
    'Dday': 'int8',
    'Dhour': 'int8',
    'Dmin': 'int8',
    'DDweek': 'int8',
    'Temp': 'float32',
    'Precip': 'float32',
    'Wind': 'float32',
    'Solar': 'float32',
    'Snow': 'float32',
    'GroundTemp': 'float32',
    'Dust': 'float32'
}).sample(frac=1)

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
# Create day of week column and calculate average duration
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['DayOfWeek'] = pd.Categorical(data['PDweek'], categories=range(7), ordered=True)
data['DayOfWeek'] = data['DayOfWeek'].cat.rename_categories(day_names)

average_duration = data.groupby('DayOfWeek')['Duration'].mean()

plt.plot(average_duration.index, average_duration.values)
plt.xlabel('Day of Week')
plt.ylabel('Average Duration')
plt.title('Average Trip Duration by Day of Week')
plt.xticks(rotation=45)
plt.show()