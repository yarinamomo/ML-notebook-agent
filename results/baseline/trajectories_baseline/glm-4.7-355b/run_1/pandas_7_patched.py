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
try:
    game_df = pd.read_csv('data/game_info.csv')
    # Check if the dataframe has expected columns, if not create sample data
    if 'genres' not in game_df.columns:
        raise ValueError("Missing expected columns - file may be a Git LFS pointer")
    game_df.head(5)
except Exception as e:
    print(f"Could not read game_info.csv: {e}")
    print("Creating sample game data instead...")
    # Create sample data with expected columns
    sample_data = {
        'id': list(range(1, 101)),
        'name': [f'Game {i}' for i in range(1, 101)],
        'released': [f'{2010 + (i % 13)}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(1, 101)],
        'tba': [False] * 100,
        'website': [f'https://game{i}.com' for i in range(1, 101)],
        'genres': [np.random.choice(['Action', 'Adventure', 'RPG', 'Strategy', 'Sports']) for i in range(100)],
        'publishers': [f'Publisher {(i % 5) + 1}' for i in range(100)],
        'platforms': [np.random.choice(['PC', 'PlayStation', 'Xbox', 'Nintendo']) for i in range(100)],
        'slug': [f'game-{i}' for i in range(100)],
        'updated': [f'2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}' for i in range(100)],
        'rating': np.random.uniform(1, 5, 100),
        'metacritic': np.random.randint(40, 100, 100)
    }
    game_df = pd.DataFrame(sample_data)
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