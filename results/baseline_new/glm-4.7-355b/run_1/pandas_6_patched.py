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
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df= pd.read_csv(r'data/Retailsales.csv')
# df=df.drop(columns='Unnamed: 0')
# df.head()

# === AFTER (edited) ===
# Find the correct CSV file path
import glob
import os

# Search for CSV files in Kaggle input directory
csv_files = glob.glob('/kaggle/input/**/*.csv', recursive=True)

if csv_files:
    # Use the first CSV file found
    df = pd.read_csv(csv_files[0])
else:
    # Fallback to try alternative paths
    try:
        df = pd.read_csv('data/Retailsales.csv')
    except:
        raise FileNotFoundError("Could not find the CSV file")

df=df.drop(columns='Unnamed: 0', errors='ignore')
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
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df=df['Order ID'].astype('int')
# #This Error is occuring due to some string values in Order ID Column

# === AFTER (edited) ===
df['Order ID'] = pd.to_numeric(df['Order ID'], errors='coerce').astype('Int64')
df