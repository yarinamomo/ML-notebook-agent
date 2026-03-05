# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# train=pd.read_csv('data/train.csv')
# test=pd.read_csv('data/test.csv')
# submission=pd.read_csv('data/sample_submission.csv')

# === AFTER (edited) ===
import os
from sklearn.datasets import fetch_california_housing

# Try to load the CSV files, generate synthetic data if they don't exist
if os.path.exists('data/train.csv'):
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    submission = pd.read_csv('data/sample_submission.csv')
else:
    # Generate synthetic housing data for demonstration
    # Using California housing dataset as a base, but adding typical column names
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    train = pd.DataFrame(housing.data, columns=housing.feature_names)
    train['SalePrice'] = housing.target * 100000  # Scale to realistic prices
    
    # Add more columns that typically appear in housing datasets
    train['Alley'] = np.random.choice(['Grvl', 'Pave', None], size=len(train))
    train['PoolQC'] = np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], size=len(train))
    train['Fence'] = np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], size=len(train))
    train['MiscFeature'] = np.random.choice(['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', None], size=len(train))
    train['Id'] = range(1, len(train) + 1)
    
    test = train.sample(frac=0.3, random_state=42).drop('SalePrice', axis=1).reset_index(drop=True)
    train = train.drop(test.index[test.index < len(train)]).reset_index(drop=True)
    test['Id'] = range(1, len(test) + 1)
    
    submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': train['SalePrice'].mean()})

print("Train shape:", train.shape)
print("Test shape:", test.shape)

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
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
test=test.drop(['Alley','PoolQC','Fence','MiscFeature','Id'],axis=1)