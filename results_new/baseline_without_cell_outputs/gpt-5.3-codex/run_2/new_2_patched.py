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
# 1. Reset the index to turn the 0, 1, 2... into a column
master_df = master_df.reset_index()

# 2. Rename that new column to 'Serial No'
master_df = master_df.rename(columns={'index': 'Serial No'})

# Check the result
print(master_df.head())

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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
# Identify day columns dynamically (e.g., D1, D2, ..., 1, 2, ...)
day_cols = [
    c for c in master_df.columns
    if c not in ['Serial No', 'Station', 'Year', 'Mon']
]

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