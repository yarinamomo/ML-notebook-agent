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

# Get only numeric columns for boxplot
numeric_df = df.select_dtypes(include=[np.number])

if len(numeric_df.columns) == 0:
    print("No numeric columns found in the dataset. Skipping boxplots.")
else:
    fig, ax = plt.subplots(len(numeric_df.columns), 1, figsize=(5, 20))
    
    # If only one subplot, wrap ax in list for consistent iteration
    if len(numeric_df.columns) == 1:
        ax = [ax]
    
    for i, col in enumerate(list(numeric_df.columns.values)):
        axes_box = ax[i]
        sns.boxplot(data=numeric_df, y=col, ax=axes_box, color='#a5c687')
        ax[i].set_title(col, fontsize=15, color='magenta')