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
import pandas as pd
import numpy as np

# Since the actual data isn't available (Git LFS pointer), create synthetic HMEQ-like data
np.random.seed(42)
n = 1000

# Create synthetic HMEQ dataset
df = pd.DataFrame({
    'BAD': np.random.choice([0, 1], size=n, p=[0.8, 0.2]),  # 20% default rate
    'LOAN': np.random.uniform(10000, 89000, n),
    'MORTDUE': np.random.uniform(20000, 200000, n),
    'VALUE': np.random.uniform(50000, 300000, n),
    'YOJ': np.random.uniform(0, 40, n),
    'DEROG': np.random.randint(0, 10, n),
    'DELINQ': np.random.randint(0, 15, n),
    'CLAGE': np.random.uniform(0, 60, n),
    'NINQ': np.random.randint(0, 10, n),
    'CLNO': np.random.randint(1, 50, n)
})

print(f"Created synthetic dataset with {n} rows")
print(f"Columns: {df.columns.tolist()}")
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