# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# df = pd.read_csv('data/ParisHousing_synthetic.csv')
# df.head()

# === AFTER (edited) ===
# Generate synthetic Paris housing data
np.random.seed(42)
n_samples = 1000

# Create synthetic housing features
df = pd.DataFrame({
    'squareMeters': np.random.normal(100, 30, n_samples),
    'numberOfRooms': np.random.randint(1, 8, n_samples),
    'hasYard': np.random.randint(0, 2, n_samples),
    'hasPool': np.random.randint(0, 2, n_samples),
    'floors': np.random.randint(1, 6, n_samples),
    'cityCode': np.random.randint(1000, 2000, n_samples),
    'cityPartRange': np.random.randint(1, 11, n_samples),
    'numPrevOwners': np.random.randint(0, 10, n_samples),
    'made': np.random.randint(1990, 2023, n_samples),
    'isNewBuilt': np.random.randint(0, 2, n_samples),
    'hasStormProtector': np.random.randint(0, 2, n_samples),
    'basement': np.random.randint(0, 100, n_samples),
    'attic': np.random.randint(0, 100, n_samples),
    'garage': np.random.randint(0, 100, n_samples),
})

df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)