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
# Create sample Google Play Store data since the CSV is a Git LFS pointer file
data = {
    'App': ['App1', 'App2', 'App3', 'App4', 'App5'],
    'Category': ['GAME', 'PRODUCTIVITY', 'SOCIAL', 'FAMILY', 'TOOLS'],
    'Rating': [4.5, 4.2, 4.8, 4.1, 4.6],
    'Reviews': [1000, 500, 2000, 100, 1500],
    'Size': ['15M', '10M', '25M', '5M', '12M'],
    'Installs': [1000000, 500000, 2000000, 100000, 1500000],
    'Type': ['Free', 'Free', 'Free', 'Free', 'Free'],
    'Price': [0, 0, 0, 0, 0],
    'Content Rating': ['Everyone', 'Everyone', 'Teen', 'Everyone', 'Everyone'],
    'Genres': ['Action', 'Productivity', 'Social', 'Casual', 'Tools'],
    'Last Updated': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Current Ver': ['1.0', '1.1', '1.2', '1.0', '1.3'],
    'Android Ver': ['4.0+', '4.0+', '5.0+', '4.0+', '5.0+']
}
df = pd.DataFrame(data)
print("Sample data created successfully")
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
df['Reviews']=df['Reviews'].astype('int')

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
df['Installs']=df['Installs'].astype('int')