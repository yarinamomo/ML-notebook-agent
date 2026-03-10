# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df= pd.read_csv(r'data/Retailsales.csv')
df=df.drop(columns='Unnamed: 0')
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df=df.dropna(how='all')
df

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df=df['Order ID'].astype('int')
# #This Error is occuring due to some string values in Order ID Column

# === AFTER (edited) ===
df = df[df['Order ID'] != 'Order ID']
df['Order ID'] = df['Order ID'].astype('int')
df