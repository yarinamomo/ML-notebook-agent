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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# import pandas as pd
# 
# # Fill missing 'genres' values with the mode (most frequent) genres
# game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])
# 
# # Verify the missing values have been filled
# game_df['genres'].isnull().sum()

# === AFTER (edited) ===
import pandas as pd


# Check if 'genres' column exists before trying to fill NA values
if 'genres' in game_df.columns:
    game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])
else:
    print(f"Warning: 'genres' column not found in DataFrame. Available columns: {list(game_df.columns)}")


game_df['genres'].isnull().sum() if 'genres' in game_df.columns else None

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# game_df = game_df.drop(['website', 'tba', 'publishers', 'platforms', 'slug','updated'], axis=1)
# game_df.head(5)

# === AFTER (edited) ===
# List of columns to drop
columns_to_drop = ['website', 'tba', 'publishers', 'platforms', 'slug', 'updated']

# Only drop columns that actually exist in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in game_df.columns]

if existing_columns_to_drop:
    game_df = game_df.drop(existing_columns_to_drop, axis=1)
else:
    print(f"Warning: None of the specified columns found to drop. Available columns: {list(game_df.columns)}")
    
game_df.head(5)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')