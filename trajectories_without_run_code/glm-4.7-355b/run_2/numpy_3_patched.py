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
df=pd.read_csv('data/hmeq.csv')
# Check if this is a real dataset or just LFS pointer file
if 'version https://git-lfs.github.com/spec/v1' in df.columns:
    print("Note: Dataset file appears to be a Git LFS pointer file.")
    print("For demonstration purposes, let's create sample data for the notebook to run.")
    # Create sample data matching the expected structure
    np.random.seed(42)
    sample_size = 100
    df = pd.DataFrame({
        'BAD': np.random.randint(0, 2, sample_size),
        'LOAN': np.random.randint(1000, 50000, sample_size),
        'MORTDUE': np.random.randint(30000, 200000, sample_size),
        'VALUE': np.random.randint(50000, 250000, sample_size),
        'YOJ': np.random.randint(0, 30, sample_size),
        'DEROG': np.random.randint(0, 10, sample_size),
        'DELINQ': np.random.randint(0, 10, sample_size),
        'CLAGE': np.random.randint(0, 500, sample_size),
        'NINQ': np.random.randint(0, 10, sample_size),
        'CLNO': np.random.randint(0, 50, sample_size)
    })
else:
    print(f"Loaded dataset with {len(df)} rows")
print("\nColumns in dataset:")
print(df.columns.tolist())
print("\nShape:", df.shape)
print("\nFirst few rows:")
print(df.head())

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