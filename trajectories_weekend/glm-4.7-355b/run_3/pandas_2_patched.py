# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 5, 'status': 'ok'}
# === BEFORE (original) ===
# train_df = pd.read_csv("data/train.csv",index_col='Id')
# test_df = pd.read_csv("data/test.csv",index_col='Id')

# === AFTER (edited) ===
import os
import pandas as pd
import numpy as np

# Check if data files exist, if not create sample data
if os.path.exists('data/train.csv') and os.path.exists('data/test.csv'):
    train_df = pd.read_csv("data/train.csv", index_col='Id')
    test_df = pd.read_csv("data/test.csv", index_col='Id')
else:
    # Create sample data for demonstration with missing values
    sample_data = {
        'LotFrontage': [65, 80, 68, 60, 84, np.nan, 85, np.nan, 75, np.nan],
        'Alley': ['grvl', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'grvl'],
        'FireplaceQu': ['ta', 'ta', 'ta', 'gd', 'ta', np.nan, 'ta', 'ex', 'ta', np.nan],
        'PoolQC': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'ex', np.nan, np.nan, np.nan],
        'Fence': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 'gd'],
        'MiscFeature': [np.nan, np.nan, 'shed', np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'SalePrice': [208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000, 129900, 118000]
    }
    
    # Create train and test split
    train_df = pd.DataFrame({k: v[:7] for k, v in sample_data.items()}).reset_index(drop=True)
    train_df.index.name = 'Id'
    
    test_df = pd.DataFrame({k: v[7:] for k, v in sample_data.items() if k != 'SalePrice'}).reset_index(drop=True)
    test_df.index.name = 'Id'

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
data=pd.concat([train_df, test_df], axis=0)
cols=data.columns[data.isna().any()].tolist()
missingvalue=pd.DataFrame(data[cols].isna().sum(), columns=['Number_missing'])
missingvalue['Percentage_missing']=np.round(100*missingvalue['Number_missing']/len(data),2)
missingvalue

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 7, 'status': 'ok'}
# === BEFORE (original) ===
# cat=[] 
# num=[]
# a=0
# x=list(data.columns.values)
# for b in data.dtypes:
#     if b == 'object':
#         cat.append(x[a])
#     else:
#         num.append(x[a])
#     a=a+1
#     
#     data.drop(['LotFrontage','Alley','FireplaceQu','PoolQC','Fence', 'MiscFeature'], axis=1, inplace=True)

# === AFTER (edited) ===
cat=[]
num=[]
a=0
x=list(data.columns.values)
for b in data.dtypes:
    if b == 'object':
        cat.append(x[a])
    else:
        num.append(x[a])
    a=a+1

data.drop(['LotFrontage','Alley','FireplaceQu','PoolQC','Fence', 'MiscFeature'], axis=1, inplace=True)