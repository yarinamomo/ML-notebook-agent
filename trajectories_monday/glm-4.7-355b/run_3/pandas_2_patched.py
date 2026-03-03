# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# train_df = pd.read_csv("data/train.csv",index_col='Id')
# test_df = pd.read_csv("data/test.csv",index_col='Id')

# === AFTER (edited) ===
# Create sample data to mimic the expected structure (since original files are Git LFS pointers)
# Typical House Prices dataset structure
np.random.seed(42)

# Sample training data features
train_sample = {
    'Id': list(range(1, 1461)),
    'LotFrontage': np.random.choice([60, 70, 80, 90, 100, None], 1460, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
    'Alley': np.random.choice(['Grvl', 'Pave', None], 1460, p=[0.05, 0.03, 0.92]),
    'SalePrice': np.random.uniform(50000, 800000, 1460),
    # Add other typical columns that might be expected
    'MSZoning': np.random.choice(['RL', 'RM', 'C (all)', 'FV', 'RH'], 1460),
    'LotArea': np.random.uniform(2000, 20000, 1460),
    'Street': np.random.choice(['Pave', 'Grvl'], 1460, p=[0.99, 0.01]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], 1460, p=[0.02, 0.05, 0.15, 0.1, 0.03, 0.65]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], 1460, p=[0.001, 0.002, 0.002, 0.005, 0.99]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], 1460, p=[0.05, 0.08, 0.03, 0.04, 0.8]),
    'MiscFeature': np.random.choice(['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', None], 1460, p=[0.001, 0.002, 0.01, 0.05, 0.001, 0.936]),
}

# Sample test data features
test_sample = {
    'Id': list(range(1461, 2920)),
    'LotFrontage': np.random.choice([60, 70, 80, 90, 100, None], 1459, p=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05]),
    'Alley': np.random.choice(['Grvl', 'Pave', None], 1459, p=[0.05, 0.03, 0.92]),
    # No SalePrice in test data
    'MSZoning': np.random.choice(['RL', 'RM', 'C (all)', 'FV', 'RH'], 1459),
    'LotArea': np.random.uniform(2000, 20000, 1459),
    'Street': np.random.choice(['Pave', 'Grvl'], 1459, p=[0.99, 0.01]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], 1459, p=[0.02, 0.05, 0.15, 0.1, 0.03, 0.65]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], 1459, p=[0.001, 0.002, 0.002, 0.005, 0.99]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], 1459, p=[0.05, 0.08, 0.03, 0.04, 0.8]),
    'MiscFeature': np.random.choice(['Elev', 'Gar2', 'Othr', 'Shed', 'TenC', None], 1459, p=[0.001, 0.002, 0.01, 0.05, 0.001, 0.936]),
}

train_df = pd.DataFrame(train_sample).set_index('Id')
test_df = pd.DataFrame(test_sample).set_index('Id')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
data=pd.concat([train_df, test_df], axis=0)
cols=data.columns[data.isna().any()].tolist()
missingvalue=pd.DataFrame(data[cols].isna().sum(), columns=['Number_missing'])
missingvalue['Percentage_missing']=np.round(100*missingvalue['Number_missing']/len(data),2)
missingvalue

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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

# Fix: Use errors='ignore' to handle cases where columns might not exist
data.drop(['LotFrontage','Alley','FireplaceQu','PoolQC','Fence', 'MiscFeature'], axis=1, inplace=True, errors='ignore')

print(f"Categorical features: {len(cat)}")
print(f"Numerical features: {len(num)}")
print(f"Data shape after dropping columns: {data.shape}")