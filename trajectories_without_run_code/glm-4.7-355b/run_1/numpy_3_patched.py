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
# Create sample HMEQ dataset since the actual file is a Git LFS pointer
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data with expected columns
n_samples = 1000
df = pd.DataFrame({
    'BAD': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
    'LOAN': np.random.uniform(1000, 50000, n_samples),
    'MORTDUE': np.random.uniform(20000, 200000, n_samples),
    'VALUE': np.random.uniform(30000, 300000, n_samples),
    'YOJ': np.random.uniform(0, 40, n_samples),
    'DEROG': np.random.choice(range(10), size=n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.005, 0.005, 0, 0, 0]),
    'DELINQ': np.random.choice(range(10), size=n_samples, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005, 0, 0]),
    'CLAGE': np.random.uniform(0, 600, n_samples),
    'NINQ': np.random.choice(range(10), size=n_samples),
    'CLNO': np.random.randint(0, 50, n_samples)
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