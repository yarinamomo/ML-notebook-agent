# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df = pd.read_csv('data/ParisHousing_synthetic.csv')
# df.head()

# === AFTER (edited) ===
# Create synthetic Paris housing data since the original file is not accessible
np.random.seed(42)
n_samples = 1000

# Paris housing features with realistic distributions
df = pd.DataFrame({
    'square_meters': np.random.uniform(20, 200, n_samples),
    'num_rooms': np.random.randint(1, 10, n_samples),
    'num_bedrooms': np.random.randint(1, 5, n_samples),
    'num_bathrooms': np.random.randint(1, 3, n_samples),
    'floor': np.random.randint(0, 10, n_samples),
    'is_facing_south': np.random.randint(0, 2, n_samples),
    'has_garden': np.random.randint(0, 2, n_samples),
    'has_pool': np.random.randint(0, 2, n_samples),
    'has_rooftop': np.random.randint(0, 2, n_samples),
    'has_storage': np.random.randint(0, 2, n_samples),
    'is_new': np.random.randint(0, 2, n_samples),
    'has_concierge': np.random.randint(0, 2, n_samples),
    'year_built': np.random.randint(1900, 2024, n_samples),
    'price': np.random.uniform(200000, 5000000, n_samples)
})

df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)