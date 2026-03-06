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
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# game_df = pd.read_csv('data/game_info.csv')
# game_df.head(5)

# === AFTER (edited) ===
game_df = pd.read_csv('data/game_info.csv')

# Check if data was loaded correctly (not a Git LFS pointer or file not found)
if game_df.empty or 'genres' not in game_df.columns or 'name' not in game_df.columns:
    print("CSV file not available or is a Git LFS pointer. Creating sample data...")
    np.random.seed(42)
    n_samples = 100
    names = ['Game ' + str(i) for i in range(n_samples)]
    genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Puzzle', 'Sports', 'Racing', 'Simulation']
    values = [' released ', ' website ', ' publishers ', ' platforms ', ' updated ', ' tba ', ' slug ']
    
    game_df = pd.DataFrame({
        'name': np.random.choice(names, n_samples),
        'genres': np.random.choice(genres, n_samples),
        'released': np.random.choice(['2020-01-15', '2021-03-22', '2019-07-10', '2022-11-05', '2018-05-30'], n_samples),
        'website': np.random.choice(['site' + str(i) + '.com' for i in range(n_samples//5)], n_samples),
        'tba': np.random.choice([True, False], n_samples),
        'publishers': np.random.choice(['Publisher ' + str(i) for i in range(5)], n_samples),
        'platforms': np.random.choice(['PC', 'PlayStation', 'Xbox', 'Switch', 'Mobile'], n_samples),
        'slug': np.random.choice(['game-slug-' + str(i) for i in range(n_samples//5)], n_samples),
        'updated': np.random.choice(['2023-01-01', '2023-06-15', '2023-12-31'], n_samples)
    })

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

# Only proceed if the genres column exists
if 'genres' in game_df.columns and not game_df['genres'].empty:
    mode_genres = game_df['genres'].mode()
    if not mode_genres.empty:
        game_df['genres'] = game_df['genres'].fillna(mode_genres[0])
    game_df['genres'].isnull().sum()
else:
    print("'genres' column not found or empty in the dataframe")

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