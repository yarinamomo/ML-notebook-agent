# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
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
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# df= pd.read_csv(r'data/Retailsales.csv')
# df=df.drop(columns='Unnamed: 0')
# df.head()

# === AFTER (edited) ===
df= pd.read_csv(r'data/Retailsales.csv')
# Drop 'Unnamed: 0' column only if it exists
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns='Unnamed: 0')
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
df=df.dropna(how='all')
df

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
# === BEFORE (original) ===
# df=df['Order ID'].astype('int')
# #This Error is occuring due to some string values in Order ID Column

# === AFTER (edited) ===
# Convert 'Order ID' column to int if it exists
if 'Order ID' in df.columns:
    df = df['Order ID'].astype('int')