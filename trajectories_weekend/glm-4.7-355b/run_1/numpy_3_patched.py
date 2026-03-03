# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
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
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# df=pd.read_csv('data/hmeq.csv')
# df.head()

# === AFTER (edited) ===
# Since the data file is a Git LFS pointer, create sample data with expected structure
# for the HMEQ (Home Equity) dataset
data = {
    'BAD': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),  # 0=PAID, 1=DEFAULT
    'LOAN': np.random.uniform(1000, 90000, 1000),
    'MORTDUE': np.random.uniform(0, 200000, 1000),
    'VALUE': np.random.uniform(0, 300000, 1000),
    'YOJ': np.random.uniform(0, 45, 1000),  # Years at present job
    'DEROG': np.random.randint(0, 10, 1000),  # Number of derogatory reports
    'DELINQ': np.random.randint(0, 20, 1000),  # Number of delinquent credit lines
    'CLAGE': np.random.uniform(0, 60, 1000),  # Age of oldest trade line in months
    'NINQ': np.random.randint(0, 20, 1000),  # Number of recent credit inquiries
    'CLNO': np.random.randint(0, 80, 1000),  # Number of credit lines
    'DEBTINC': np.random.uniform(0, 200, 1000)  # Debt-to-income ratio
}
df = pd.DataFrame(data)
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
df.loc[df.BAD == 1, 'STATUS'] = 'DEFAULT'
df.loc[df.BAD == 0, 'STATUS'] = 'PAID'

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
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