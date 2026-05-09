# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
submission=pd.read_csv('data/sample_submission.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# test=test.drop(['Alley','PoolQC','Fence','MiscFeature','Id'],axis=1)

# === AFTER (edited) ===
test = test.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], axis=1, errors='ignore')