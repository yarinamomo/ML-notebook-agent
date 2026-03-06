# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df=pd.read_csv('data/3711.csv');df

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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

# Get only numeric columns that can be plotted in a boxplot
numeric_cols = df.select_dtypes(include=[np.number]).columns

for i, col in enumerate(numeric_cols):
    axes_box  = ax[i]
    sns.boxplot(data=df, x=col, ax=axes_box,color='#a5c687')
    ax[i].set_title(col,fontsize=15,color='magenta')

# Hide any unused subplots if there are fewer than 6 numeric columns
for i in range(len(numeric_cols), len(ax)):
    ax[i].set_visible(False)