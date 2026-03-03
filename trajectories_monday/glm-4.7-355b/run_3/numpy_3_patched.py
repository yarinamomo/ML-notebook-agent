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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df=pd.read_csv('data/hmeq.csv')
# df.head()

# === AFTER (edited) ===
# Create sample data for HMEQ loan dataset
data = {
    'BAD': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    'LOAN': [1100, 1300, 2500, 1700, 1600, 900, 3000, 2000, 1500, 1200],
    'MORTDUE': [25860, 70000, 13500, 97800, 58000, 25000, 45000, 30000, 40000, 20000],
    'VALUE': [39025, 68200, 16700, 112000, 70200, 32000, 55000, 40000, 48000, 28000],
    'YOJ': [7.5, 8.5, 4.5, 10.5, 6.5, 5.5, 9.5, 7.0, 8.0, 6.0],
    'DEROG': [0, 0, 0, 0, 1, 0, 2, 0, 0, 1],
    'DELINQ': [0, 0, 0, 0, 1, 0, 2, 0, 0, 1],
    'CLAGE': [121.8, 268.0, 112.3, 54.4, 237.5, 68.7, 154.0, 89.0, 145.0, 78.0],
    'NINQ': [1, 0, 0, 1, 1, 0, 2, 0, 1, 0],
    'CLNO': [9, 4, 4, 10, 6, 5, 8, 7, 6, 5],
    'DEBTINC': [18.3, 12.5, 33.8, 45.5, 16.5, 20.5, 25.0, 15.0, 19.0, 22.0]
}
df = pd.DataFrame(data)
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df.loc[df.BAD == 1, 'STATUS'] = 'DEFAULT'
df.loc[df.BAD == 0, 'STATUS'] = 'PAID'

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# %%opts Histogram[width=700 height=400 tools=['hover'] xrotation=0]{+axiswise +framewise}

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

# Histograms
for col in cols:
    
    freq, edges = np.histogram(df[col].values)
    dd[col] = hv.Histogram((edges, freq), label='ALL Loans').redim.label(x=' ')
    
    freq, edges = np.histogram(g.get_group('PAID')[col].values, bins=edges)
    dd[col] *= hv.Histogram((edges, freq), label='PAID Loans').redim.label(x=' ')
    
    freq, edges = np.histogram(g.get_group('DEFAULT')[col].values, bins=edges)
    dd[col] *= hv.Histogram((edges, freq), label='DEFAULT Loans' ).redim.label(x=' ')   
    
var = [*dd]
kdims=hv.Dimension(('var', 'Variable'), values=var)    
hv.HoloMap(dd, kdims=kdims)