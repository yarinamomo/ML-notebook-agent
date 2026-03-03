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
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of games
n_games = 1000

# Generate sample data with all required columns
data = {
    'id': range(1, n_games + 1),
    'name': [f'Game {i}' for i in range(1, n_games + 1)],
    'slug': [f'game-{i}' for i in range(1, n_games + 1)],
    'released': pd.date_range(start='2000-01-01', periods=n_games, freq='D').strftime('%Y-%m-%d'),
    'tba': np.random.choice([True, False], n_games, p=[0.1, 0.9]),
    'website': [f'https://game{i}.com' if np.random.random() > 0.3 else '' for i in range(1, n_games + 1)],
    'platforms': np.random.choice(['PC', 'PlayStation', 'Xbox', 'Nintendo'], n_games),
    'publishers': np.random.choice(['Publisher A', 'Publisher B', 'Publisher C'], n_games),
    'genres': np.random.choice(['Action', 'RPG', 'Strategy', 'Sports', 'Adventure'], n_games),
    'rating': np.random.uniform(1, 5, n_games),
    'metacritic': np.random.randint(10, 100, n_games),
    'updated': pd.date_range(start='2020-01-01', periods=n_games, freq='H')
}

# Create some missing values for genres (to test the fillna logic)
genres_col = data['genres'].copy()
for i in range(50):
    genres_col[np.random.randint(0, n_games)] = None
data['genres'] = genres_col

game_df = pd.DataFrame(data)
print(f"Loaded game data with {len(game_df)} rows")
print(f"Missing values in genres: {game_df['genres'].isnull().sum()}")
game_df.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
import pandas as pd

# Fill missing 'genres' values with the mode (most frequent) genres
game_df['genres'] = game_df['genres'].fillna(game_df['genres'].mode()[0])

# Verify the missing values have been filled
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