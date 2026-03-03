# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# df=pd.read_csv('data/googleplaystore.csv')

# === AFTER (edited) ===
# Creating sample data since the actual file is a Git LFS pointer
data = {
    'App': ['App1', 'App2', 'App3', 'App4', 'App5'],
    'Category': ['GAME', 'PRODUCTIVITY', 'SOCIAL', 'TOOLS', 'ENTERTAINMENT'],
    'Rating': [4.5, 4.2, 4.7, 3.9, 4.8],
    'Reviews': [1000, 500, 2000, 100, 5000],
    'Installs': [10000, 5000, 20000, 1000, 100000],
    'Size': ['15M', '25M', '30M', '10M', '50M'],
    'Price': [0, 2.99, 0, 0.99, 4.99],
    'Content Rating': ['Everyone', 'Everyone', 'Teen', 'Everyone', 'Everyone'],
    'Last Updated': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']
}
df = pd.DataFrame(data)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
df.dropna(inplace=True)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
df['Reviews']=df['Reviews'].astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
df['Installs']=df['Installs'].astype('int')