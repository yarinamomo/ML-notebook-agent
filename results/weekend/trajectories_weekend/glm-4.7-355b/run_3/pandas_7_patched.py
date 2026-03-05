# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
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
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# game_df = pd.read_csv('data/game_info.csv')
# game_df.head(5)

# === AFTER (edited) ===
# Create sample game data since the actual file is a Git LFS pointer
game_df = pd.DataFrame({
    'id': range(1, 101),
    'name': [f'Game {i}' for i in range(1, 101)],
    'released': [f'{2020 + (i % 6)}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(100)],
    'website': [f'https://game{i}.com' if i % 3 == 0 else None for i in range(100)],
    'tba': [False] * 100,
    'genres': [['Action', 'Adventure'] if i % 2 == 0 else ['RPG'] if i % 3 == 0 else ['Strategy'] for i in range(100)],
    'publishers': [f'Publisher {i % 5}' for i in range(100)],
    'platforms': [['PC', 'PlayStation'] if i % 2 == 0 else ['Xbox'] for i in range(100)],
    'slug': [f'game-{i}' for i in range(100)],
    'updated': [f'2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(100)]
})

# Make some genres None for testing fillna logic
game_df.loc[game_df.index[::10], 'genres'] = None

game_df.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
# === BEFORE (original) ===
# import pandas as pd
# 
# # Fill missing 'genres' values with the mode (most frequent) genres
# game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])
# 
# # Verify the missing values have been filled
# game_df['genres'].isnull().sum()

# === AFTER (edited) ===
# For genres containing lists, fill None with a simple list
default_genre = ['Action']

# Create a new list of genres with None replaced
game_df['genres'] = game_df['genres'].apply(lambda x: default_genre if x is None else x)

game_df['genres'].isnull().sum()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
game_df = game_df.drop(['website', 'tba', 'publishers', 'platforms', 'slug','updated'], axis=1)
game_df.head(5)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')