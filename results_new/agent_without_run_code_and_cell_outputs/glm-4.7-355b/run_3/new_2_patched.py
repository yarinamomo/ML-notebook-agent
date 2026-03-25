# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import os

# Define the directory path
directory = 'data/'

# Create a dictionary to store the dataframes
dataframes = {}

# Loop through every file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        # Create a clean key name (e.g., 'Dhaka_wind' instead of 'Dhaka_wind - Dhaka.csv')
        # This splits by the dash and strips whitespace
        key_name = filename.split('-')[0].strip()
        
        # Construct the full file path
        file_path = os.path.join(directory, filename)
        
        # Read the CSV and store it in the dictionary
        dataframes[key_name] = pd.read_csv(file_path)
        
        print(f"Loaded: {key_name}")

# Example: Accessing a specific dataframe
print(dataframes['Dhaka_wind'].head())

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# 1. Combine all dataframes from the dictionary into a single list
# We add a 'Station' column to each one before merging
df_list = []
for station_name, df in dataframes.items():
    # Insert 'Station' as the first column
    df.insert(0, 'Station', station_name)
    df_list.append(df)

# 2. Concatenate all dataframes into one master dataframe
master_df = pd.concat(df_list, ignore_index=True)

# 3. Check the result
print(f"Merged Shape: {master_df.shape}")
print(master_df.head())

# Optional: Save the merged data to a CSV
# master_df.to_csv('combined_wind_data.csv', index=False)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# 1. Reset the index to turn the 0, 1, 2... into a column
master_df = master_df.reset_index()

# 2. Rename that new column to 'Serial No'
master_df = master_df.rename(columns={'index': 'Serial No'})

# Check the result
print(master_df.head())

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# # Create the long-form dataframe
# df_long = master_df.melt(id_vars=['Station', 'Year', 'Mon'], 
#                          value_vars=day_cols, 
#                          var_name='Day_Raw', 
#                          value_name='WindSpeed')
# 
# # Clean WindSpeed
# df_long['WindSpeed'] = df_long['WindSpeed'].replace('CLM', 0)
# df_long['WindSpeed'] = pd.to_numeric(df_long['WindSpeed'], errors='coerce')
# df_long = df_long.dropna(subset=['WindSpeed']).sort_values(['Station', 'Year', 'Mon'])
# 
# # Define your scaler here as well
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaler.fit(df_long[['WindSpeed']])

# === AFTER (edited) ===
# Get all columns that could be day columns (excluding Station, Serial No, Year, Mon)
day_cols = [col for col in master_df.columns if col not in ['Station', 'Serial No', 'Year', 'Mon']]

df_long = master_df.melt(id_vars=['Station', 'Year', 'Mon'],
                         value_vars=day_cols,
                         var_name='Day_Raw',
                         value_name='WindSpeed')


df_long['WindSpeed'] = df_long['WindSpeed'].replace('CLM', 0)
df_long['WindSpeed'] = pd.to_numeric(df_long['WindSpeed'], errors='coerce')
df_long = df_long.dropna(subset=['WindSpeed']).sort_values(['Station', 'Year', 'Mon'])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_long[['WindSpeed']])