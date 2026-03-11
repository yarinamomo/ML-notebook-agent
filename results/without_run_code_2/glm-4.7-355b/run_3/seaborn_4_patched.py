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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv('data/ParisHousing_synthetic.csv')
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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
# Get the number of columns in the dataframe
num_cols = len(df.columns)
num_cols_subplot = 7
num_rows_subplot = (num_cols + num_cols_subplot - 1) // num_cols_subplot

fig, ax = plt.subplots(ncols=num_cols_subplot, nrows=num_rows_subplot, figsize=(20, num_rows_subplot * 5))
ax = ax.flatten()

for index, col in enumerate(df.columns):
    sns.boxplot(y=col, data=df, ax=ax[index])

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)