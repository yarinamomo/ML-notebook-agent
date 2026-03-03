# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df=pd.read_csv('data/googleplaystore.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df.dropna(inplace=True)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df['Reviews']=df['Reviews'].astype('int')

# === AFTER (edited) ===
df['Reviews'] = df['Reviews'].apply(lambda x: float(str(x).replace('M', '')) * 1e6 if 'M' in str(x) else float(str(x).replace('k', '')) * 1e3 if 'k' in str(x) else float(x)).astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df['Installs']=df['Installs'].astype('int')

# === AFTER (edited) ===
df['Installs'] = df['Installs'].astype(str).str.replace('+', '').str.replace(',', '').astype('int')