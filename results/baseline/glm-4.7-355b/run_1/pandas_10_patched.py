# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df=pd.read_csv('data/googleplaystore.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df.dropna(inplace=True)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df['Reviews']=df['Reviews'].astype('int')

# === AFTER (edited) ===
# Strip whitespace from column names and convert to title case
df.columns = df.columns.str.strip().str.title()

# Check column names
print("Available columns:", df.columns.tolist())

# Now attempt the conversion
df['Reviews']=df['Reviews'].astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df['Installs']=df['Installs'].astype('int')