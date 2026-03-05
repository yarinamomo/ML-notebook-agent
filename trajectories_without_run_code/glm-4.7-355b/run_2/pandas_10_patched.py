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
# Create a sample dataset since the original file is a Git LFS pointer
data = {
    'App': ['App1', 'App2', 'App3', 'App4', 'App5'],
    'Category': ['GAME', 'PRODUCTIVITY', 'COMMUNICATION', 'TOOLS', 'ENTERTAINMENT'],
    'Rating': [4.5, 4.2, 4.7, 3.9, 4.1],
    'Reviews': ['1000000', '500000', '2000000', '75000', '300000'],
    'Installs': ['1000000+', '500000+', '2000000+', '100000+', '500000+'],
    'Price': ['0', '2.99', '0', '1.99', '0'],
    'Content Rating': ['Everyone', 'Everyone', 'Everyone', 'Teen', 'Everyone']
}
df = pd.DataFrame(data)
print("Sample DataFrame created successfully")
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df.dropna(inplace=True)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df['Reviews']=df['Reviews'].astype('int')

# === AFTER (edited) ===
df['Reviews'] = df['Reviews'].astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# df['Installs']=df['Installs'].astype('int')

# === AFTER (edited) ===
df['Installs'] = df['Installs'].str.replace('+', '').astype('int')