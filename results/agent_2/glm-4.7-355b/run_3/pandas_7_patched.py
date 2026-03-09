# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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
# execution_status: {'status': 'not run'}
game_df = pd.read_csv('data/game_info.csv')
game_df.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import pandas as pd

# Fill missing 'genres' values with the mode (most frequent) genres
game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])

# Verify the missing values have been filled
game_df['genres'].isnull().sum()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
game_df = game_df.drop(['website', 'tba', 'publishers', 'platforms', 'slug','updated'], axis=1)
game_df.head(5)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')

# === AFTER (edited) ===
# Handle NaN values in released column before converting to year
game_df['released'] = pd.to_numeric(game_df['released'].apply(lambda x: str(x).split('-')[0]), errors='coerce')
# Fill NaN years with 0 or keep them as NaN
game_df['released'].fillna(0, inplace=True)
# Convert to integer
game_df['released'] = game_df['released'].astype('int')