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
# Create sample HMEQ dataset since the original file is a Git LFS pointer
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'BAD': np.random.choice([0, 1], n_samples, p=[0.75, 0.25]),
    'LOAN': np.random.uniform(1000, 90000, n_samples),
    'MORTDUE': np.random.uniform(20000, 400000, n_samples),
    'VALUE': np.random.uniform(30000, 600000, n_samples),
    'YOJ': np.random.uniform(0, 40, n_samples),
    'DEROG': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.8, 0.1, 0.05, 0.03, 0.02]),
    'DELINQ': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.03]),
    'CLAGE': np.random.uniform(0, 600, n_samples),
    'NINQ': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.5, 0.25, 0.15, 0.07, 0.03]),
    'CLNO': np.random.randint(1, 50, n_samples),
    'DEBTINC': np.random.uniform(0, 100, n_samples)
})

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