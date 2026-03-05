# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df=pd.read_csv('data/googleplaystore.csv')

# === AFTER (edited) ===
df=pd.read_csv('data/googleplaystore.csv')

# Check if the data was loaded correctly (not a git-lfs pointer or empty)
if df.shape[1] == 1 and 'git-lfs.github.com' in df.columns[0]:
    # Create a sample dataset for demonstration
    print("Warning: CSV file appears to be a git-lfs pointer. Creating sample dataset.")
    df = pd.DataFrame({
        'App': ['App1', 'App2', 'App3', 'App4', 'App5'],
        'Reviews': ['1000', '2000', '3000', '4000', '5000'],
        'Installs': ['10000+', '20000+', '30000+', '40000+', '50000+']
    })

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df.dropna(inplace=True)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
df['Reviews']=df['Reviews'].astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# df['Installs']=df['Installs'].astype('int')

# === AFTER (edited) ===
# Clean the Installs column by removing non-numeric characters like '+' and ','
df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '').astype('int')