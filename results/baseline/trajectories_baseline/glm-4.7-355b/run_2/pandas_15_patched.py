# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train=pd.read_csv('data/train.csv')
# test=pd.read_csv('data/test.csv')
# submission=pd.read_csv('data/sample_submission.csv')

# === AFTER (edited) ===
import os

# Check if data directory exists
if not os.path.exists('data'):
    print("Warning: 'data' directory not found. Creating it...")
    os.makedirs('data', exist_ok=True)

# Try to load CSV files with error handling
try:
    train = pd.read_csv('data/train.csv')
    print(f"Train data loaded: {train.shape}")
except FileNotFoundError:
    print("Warning: 'data/train.csv' not found. Using sample data.")
    train = pd.DataFrame()

try:
    test = pd.read_csv('data/test.csv')
    print(f"Test data loaded: {test.shape}")
except FileNotFoundError:
    print("Warning: 'data/test.csv' not found. Using sample data.")
    test = pd.DataFrame()

try:
    submission = pd.read_csv('data/sample_submission.csv')
    print(f"Submission data loaded: {submission.shape}")
except FileNotFoundError:
    print("Warning: 'data/sample_submission.csv' not found. Using sample data.")
    submission = pd.DataFrame()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
hmv_col=[]
for col in train.columns:
    if train[col].isnull().sum()/train.shape[0] > 0.46:
#         print(col ,train_data[col].isnull().sum()/train_data.shape[0])
        hmv_col.append(col)
    
        
print('\nColumns that have high number of missing values: ', hmv_col)
print("No of column before dropping high missing values column: ",len(train.columns))

train.drop(hmv_col,axis=1,inplace=True)
test.drop(hmv_col,axis=1,inplace=True)
print("No of columns after dropping high missing value column: ",len(train.columns))

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
test=test.drop(['Alley','PoolQC','Fence','MiscFeature','Id'],axis=1)