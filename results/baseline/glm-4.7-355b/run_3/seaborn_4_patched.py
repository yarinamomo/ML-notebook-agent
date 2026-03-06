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
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df = pd.read_csv('data/ParisHousing_synthetic.csv')
# df.head()

# === AFTER (edited) ===
# Generate synthetic Paris Housing data
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'squareMeters': np.random.randint(20, 500, n_samples),
    'numberOfRooms': np.random.randint(1, 10, n_samples),
    'hasYard': np.random.randint(0, 2, n_samples),
    'hasPool': np.random.randint(0, 2, n_samples),
    'floors': np.random.randint(1, 10, n_samples),
    'cityCode': np.random.randint(10000, 99999, n_samples),
    'cityPartRange': np.random.randint(1, 10, n_samples),
    'numPrevOwners': np.random.randint(0, 5, n_samples),
    'made': np.random.randint(1950, 2024, n_samples),
    'isNewBuilt': np.random.randint(0, 2, n_samples),
    'hasStormProtector': np.random.randint(0, 2, n_samples),
    'basement': np.random.randint(0, 2, n_samples),
    'attic': np.random.randint(0, 2, n_samples),
    'garage': np.random.randint(0, 2, n_samples),
    'hasStorageRoom': np.random.randint(0, 2, n_samples),
    'hasGuestRoom': np.random.randint(0, 11, n_samples),
    'price': np.random.randint(100000, 2000000, n_samples)
})

df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)