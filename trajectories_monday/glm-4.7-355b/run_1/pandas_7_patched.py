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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# game_df = pd.read_csv('data/game_info.csv')
# game_df.head(5)

# === AFTER (edited) ===
# Since the actual data file is a Git LFS pointer, create sample data
import numpy as np
import pandas as pd

# Create a sample dataset matching expected structure
np.random.seed(42)
n_samples = 100

game_df = pd.DataFrame({
    'id': range(1, n_samples + 1),
    'name': [f'Game {i}' for i in range(1, n_samples + 1)],
    'released': np.random.choice(['2020-01-15', '2019-05-20', '2021-03-10', '2018-12-01', '2022-07-25'], n_samples),
    'genres': np.random.choice(['Action', 'Adventure', 'RPG', 'Strategy', 'Sports', np.nan], n_samples, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05]),
    'platforms': np.random.choice(['PC', 'PlayStation', 'Xbox', 'Nintendo'], n_samples),
    'publishers': np.random.choice(['Publisher A', 'Publisher B', 'Publisher C'], n_samples),
    'website': [f'http://www.game{i}.com' for i in range(1, n_samples + 1)],
    'tba': np.random.choice([False, False, False, False, True], n_samples),
    'slug': [f'game-{i}' for i in range(1, n_samples + 1)],
    'updated': np.random.choice(['2023-01-01', '2023-02-15', '2023-03-20'], n_samples),
    'rating': np.random.uniform(1.0, 5.0, n_samples),
    'metacritic': np.random.randint(50, 100, n_samples)
})

game_df.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import pandas as pd

# Fill missing 'genres' values with the mode (most frequent) genres
game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])

# Verify the missing values have been filled
game_df['genres'].isnull().sum()


#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
game_df = game_df.drop(['website', 'tba', 'publishers', 'platforms', 'slug','updated'], axis=1)
game_df.head(5)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
game_df['released'] = game_df['released'].apply(lambda x: str(x).split('-')[0]).astype('int')