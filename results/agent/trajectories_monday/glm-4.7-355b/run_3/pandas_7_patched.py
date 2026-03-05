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
import pandas as pd
import numpy as np

# Create a mock dataframe with the expected structure for game data
np.random.seed(42)
n_rows = 100

data = {
    'name': [f'Game_{i}' for i in range(n_rows)],
    'released': np.random.choice([f'{year}-01-01' for year in range(1990, 2024)], size=n_rows),
    'tba': np.random.choice([True, False], size=n_rows),
    'website': [f'https://game{i}.com' for i in range(n_rows)],
    'genres': np.random.choice(['Action', 'Adventure', 'RPG', 'Strategy', 'Indie', 'Casual'], size=n_rows),
    'publishers': np.random.choice(['Publisher_A', 'Publisher_B', 'Publisher_C'], size=n_rows),
    'platforms': np.random.choice(['PC', 'PlayStation', 'Xbox', 'Switch'], size=n_rows),
    'slug': [f'game-{i}' for i in range(n_rows)],
    'updated': [f'{2023}-01-01' for i in range(n_rows)],
    'rating': np.random.uniform(1.0, 5.0, size=n_rows)
}

game_df = pd.DataFrame(data)
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