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
import pandas as pd

# Check what columns are available
print("Available columns in game_df:")
print(game_df.columns.tolist())
print("\nFirst few rows:")
print(game_df.head())

# Only proceed if 'genres' column exists
if 'genres' in game_df.columns:
    game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])
    game_df['genres'].isnull().sum()
else:
    print("\nColumn 'genres' not found in the dataset")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
game_df = game_df.drop(['website', 'tba', 'publishers', 'platforms', 'slug','updated'], axis=1)
game_df.head(5)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')