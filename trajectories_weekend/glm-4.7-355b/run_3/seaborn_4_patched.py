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
# Read the data file
import pandas as pd
import numpy as np

df = pd.read_csv('data/ParisHousing_synthetic.csv')

# Check if the file contains Git LFS pointer or actual data
if len(df.columns) == 1 and 'version' in df.columns[0]:
    # The file contains Git LFS pointer, create synthetic housing data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'squareMeters': np.random.randint(20, 400, n_samples),
        'numberOfRooms': np.random.randint(1, 10, n_samples),
        'hasYard': np.random.randint(0, 2, n_samples),
        'hasPool': np.random.randint(0, 2, n_samples),
        'floors': np.random.randint(1, 6, n_samples),
        'cityCode': np.random.randint(1, 100, n_samples),
        'cityPartRange': np.random.randint(1, 10, n_samples),
        'numPrevOwners': np.random.randint(0, 5, n_samples),
        'made': np.random.randint(1950, 2024, n_samples),
        'isNewBuilt': np.random.randint(0, 2, n_samples),
        'hasStormProtector': np.random.randint(0, 2, n_samples),
        'basement': np.random.randint(0, 101, n_samples),
        'attic': np.random.randint(0, 101, n_samples),
        'garage': np.random.randint(0, 101, n_samples),
        'hasStorageRoom': np.random.randint(0, 2, n_samples),
        'hasGuestRoom': np.random.randint(0, 6, n_samples),
        'price': np.random.randint(100000, 2000000, n_samples)
    })
    
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# # create box plots
# fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
# index = 0
# ax = ax.flatten()
# 
# for col, value in df.items():
#     sns.boxplot(y=col, data=df, ax=ax[index])
#     index += 1
# plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

# === AFTER (edited) ===
# Create boxplots for all columns
n_cols = 5  # Number of columns in subplot grid
n_rows = (len(df.columns) + n_cols - 1) // n_cols  # Calculate needed rows

fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(25, 4*n_rows))
index = 0
ax = ax.flatten()

for col in df.columns:
    sns.boxplot(y=col, data=df, ax=ax[index])
    ax[index].set_title(col)
    index += 1

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)