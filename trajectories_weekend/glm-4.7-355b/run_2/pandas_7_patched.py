# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
game_df = pd.read_csv('data/game_info.csv')
game_df.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import pandas as pd
# 
# # Fill missing 'genres' values with the mode (most frequent) genres
# game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])
# 
# # Verify the missing values have been filled
# game_df['genres'].isnull().sum()

# === AFTER (edited) ===
# Check if 'genres' column exists before trying to fill missing values
if 'genres' in game_df.columns:
    game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])
else:
    print("Column 'genres' not found in dataframe. Skipping fillna.")
# Display the null count for 'genres' or a warning if missing
if 'genres' in game_df.columns:
    game_df['genres'].isnull().sum()
else:
    print("'genres' column missing; null count not available.")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# game_df = game_df.drop(['website', 'tba', 'publishers', 'platforms', 'slug','updated'], axis=1)
# game_df.head(5)

# === AFTER (edited) ===
# Drop columns that exist in the dataframe
columns_to_drop = ['website', 'tba', 'publishers', 'platforms', 'slug', 'updated']
existing_columns_to_drop = [col for col in columns_to_drop if col in game_df.columns]
game_df = game_df.drop(existing_columns_to_drop, axis=1)
game_df.head(5)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')

# === AFTER (edited) ===
# Check if 'released' column exists before processing
if 'released' in game_df.columns:
    game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')
else:
    print("Column 'released' not found in dataframe")