# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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
# execution_status: {'status': 'not run'}
df = pd.read_csv('data/ParisHousing_synthetic.csv')
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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
fig, ax = plt.subplots(ncols=9, nrows=2, figsize=(20, 10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)