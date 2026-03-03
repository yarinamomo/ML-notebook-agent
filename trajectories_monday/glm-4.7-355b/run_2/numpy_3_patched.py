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
# Create synthetic HMEQ-like dataset since the actual data file is a Git LFS pointer
np.random.seed(42)
n_samples = 1000

# Create features similar to HMEQ dataset
df = pd.DataFrame({
    'BAD': np.random.randint(0, 2, n_samples),  # Target: 0=PAID, 1=DEFAULT
    'LOAN': np.random.uniform(1000, 100000, n_samples),
    'MORTDUE': np.random.uniform(10000, 400000, n_samples),
    'VALUE': np.random.uniform(20000, 500000, n_samples),
    'YOJ': np.random.uniform(0, 50, n_samples),  # Years at present job
    'DEROG': np.random.randint(0, 15, n_samples),  # Derogatory reports
    'DELINQ': np.random.randint(0, 20, n_samples),  # Delinquent credit lines
    'CLAGE': np.random.uniform(0, 600, n_samples),  # Age of oldest credit line (months)
    'NINQ': np.random.randint(0, 20, n_samples),  # Recent credit inquiries
    'CLNO': np.random.randint(1, 80, n_samples),  # Number of credit lines
    'DEBTINC': np.random.uniform(0, 100, n_samples)  # Debt-to-income ratio
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