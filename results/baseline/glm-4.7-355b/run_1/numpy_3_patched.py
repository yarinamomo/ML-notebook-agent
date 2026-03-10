# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns

import holoviews as hv
hv.extension('bokeh', 'matplotlib', logo=False)

# Avoid warnings to show up (trick for the final notebook on kaggle)
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df=pd.read_csv('data/hmeq.csv')
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df.loc[df.BAD == 1, 'STATUS'] = 'DEFAULT'
df.loc[df.BAD == 0, 'STATUS'] = 'PAID'

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# # %%opts Histogram[width=700 height=400 tools=['hover'] xrotation=0]{+axiswise +framewise}
# 
# g = df.groupby('STATUS')
# 
# cols = ['LOAN',
#         'MORTDUE', 
#         'VALUE',
#         'YOJ',
#         'DEROG',
#         'DELINQ',
#         'CLAGE',
#         'NINQ',
#         'CLNO']
# dd={}
# 
# # Histograms
# for col in cols:
#     
#     freq, edges = np.histogram(df[col].values)
#     dd[col] = hv.Histogram((edges, freq), label='ALL Loans').redim.label(x=' ')
#     
#     freq, edges = np.histogram(g.get_group('PAID')[col].values, bins=edges)
#     dd[col] *= hv.Histogram((edges, freq), label='PAID Loans').redim.label(x=' ')
#     
#     freq, edges = np.histogram(g.get_group('DEFAULT')[col].values, bins=edges)
#     dd[col] *= hv.Histogram((edges, freq), label='DEFAULT Loans' ).redim.label(x=' ')   
#     
# var = [*dd]
# kdims=hv.Dimension(('var', 'Variable'), values=var)    
# hv.HoloMap(dd, kdims=kdims)

# === AFTER (edited) ===
g = df.groupby('STATUS')

cols = ['LOAN',
        'MORTDUE',
        'VALUE',
        'YOJ',
        'DEROG',
        'DELINQ',
        'CLAGE',
        'NINQ',
        'CLNO']
dd={}


for col in cols:
    # Remove NaN values from the column before histogramming
    all_values = df[col].values[~np.isnan(df[col].values)]
    freq, edges = np.histogram(all_values)
    dd[col] = hv.Histogram((edges, freq), label='ALL Loans').redim.label(x=' ')

    paid_values = g.get_group('PAID').loc[:, col].values[~np.isnan(g.get_group('PAID').loc[:, col].values)]
    freq, edges = np.histogram(paid_values, bins=edges)
    dd[col] *= hv.Histogram((edges, freq), label='PAID Loans').redim.label(x=' ')

    default_values = g.get_group('DEFAULT').loc[:, col].values[~np.isnan(g.get_group('DEFAULT').loc[:, col].values)]
    freq, edges = np.histogram(default_values, bins=edges)
    dd[col] *= hv.Histogram((edges, freq), label='DEFAULT Loans' ).redim.label(x=' ')

var = [*dd]
kdims=hv.Dimension(('var', 'Variable'), values=var)
hv.HoloMap(dd, kdims=kdims)