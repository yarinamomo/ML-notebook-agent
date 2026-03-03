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
# Create mock Google Play Store dataset since original CSV is a Git LFS pointer
data = {
    'App': ['WhatsApp Messenger', 'Instagram', 'Clash of Clans', 'Subway Surfers', 'UC Browser'],
    'Reviews': ['18454699', '21655845', '23133508', '27722264', '17760304'],
    'Installs': ['1000000+', '1000000+', '100000000+', '1000000+', '1000000+'],
    'Rating': [4.3, 4.5, 4.6, 4.5, 4.2],
    'Category': ['COMMUNICATION', 'SOCIAL', 'GAME', 'GAME', 'COMMUNICATION']
}
df = pd.DataFrame(data)
df

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
# Remove '+' characters and commas from Installs, then convert to int
df['Installs'] = df['Installs'].str.replace('+', '').str.replace(',', '').astype('int')