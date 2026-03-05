# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# for visuallization
import plotly.express as px
# import kaleido # comment out for reproducing purposes
import scipy
from mpl_toolkits.mplot3d import Axes3D
from yellowbrick.cluster import KElbowVisualizer
# from kneed import KneeLocator

# %notebook matplotlib
from warnings import filterwarnings
filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 2, 'status': 'ok'}
df=pd.read_csv('data/3711.csv');df

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
# === BEFORE (original) ===
# sns.set_style('darkgrid')
# fig, ax = plt.subplots(6, 1, figsize=(5, 20))#Original was 9,1
# 
# for i, col in enumerate(list(df.columns.values)):
#     axes_box  = ax[i]
#     sns.boxplot(data=df, x=col, ax=axes_box,color='#a5c687')
#     ax[i].set_title(col,fontsize=15,color='magenta')

# === AFTER (edited) ===
sns.set_style('darkgrid')
fig, ax = plt.subplots(6, 1, figsize=(5, 20))

for i, col in enumerate(list(df.columns.values)):
    axes_box = ax[i]
    # Only plot if column is numeric
    if pd.api.types.is_numeric_dtype(df[col]):
        sns.boxplot(data=df, y=col, ax=axes_box, color='#a5c687')
    else:
        # Skip non-numeric columns and leave empty
        axes_box.text(0.5, 0.5, f'Non-numeric column: {col}', 
                      ha='center', va='center', transform=axes_box.transAxes)
    ax[i].set_title(col, fontsize=15, color='magenta')