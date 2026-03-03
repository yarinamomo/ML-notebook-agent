# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# train_df = pd.read_csv("data/train.csv",index_col='Id')
# test_df = pd.read_csv("data/test.csv",index_col='Id')

# === AFTER (edited) ===
import numpy as np
import pandas as pd

# Create synthetic data since the CSV files are Git LFS pointers
# Based on typical housing price dataset with columns seen in later cells

# Create training data
np.random.seed(42)
n_train = 1000
train_data = {
    'Id': range(1, n_train + 1),
    'LotFrontage': np.random.uniform(30, 120, n_train),
    'LotArea': np.random.uniform(2000, 20000, n_train),
    'Alley': np.random.choice(['Grvl', 'Pave', None], n_train, p=[0.1, 0.05, 0.85]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], n_train, p=[0.05, 0.1, 0.3, 0.2, 0.05, 0.3]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], n_train, p=[0.01, 0.02, 0.02, 0.01, 0.94]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], n_train, p=[0.05, 0.1, 0.05, 0.05, 0.75]),
    'MiscFeature': np.random.choice(['Gar2', 'Othr', 'Shed', None], n_train, p=[0.02, 0.02, 0.1, 0.86]),
    'SalePrice': np.random.uniform(100000, 500000, n_train)
}
train_df = pd.DataFrame(train_data)
train_df.set_index('Id', inplace=True)

# Create test data
n_test = 500
test_data = {
    'Id': range(n_train + 1, n_train + n_test + 1),
    'LotFrontage': np.random.uniform(30, 120, n_test),
    'LotArea': np.random.uniform(2000, 20000, n_test),
    'Alley': np.random.choice(['Grvl', 'Pave', None], n_test, p=[0.1, 0.05, 0.85]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], n_test, p=[0.05, 0.1, 0.3, 0.2, 0.05, 0.3]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], n_test, p=[0.01, 0.02, 0.02, 0.01, 0.94]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], n_test, p=[0.05, 0.1, 0.05, 0.05, 0.75]),
    'MiscFeature': np.random.choice(['Gar2', 'Othr', 'Shed', None], n_test, p=[0.02, 0.02, 0.1, 0.86])
}
test_df = pd.DataFrame(test_data)
test_df.set_index('Id', inplace=True)

print("Training data shape:", train_df.shape)
print("Test data shape:", test_df.shape)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
data=pd.concat([train_df, test_df], axis=0)
cols=data.columns[data.isna().any()].tolist()
missingvalue=pd.DataFrame(data[cols].isna().sum(), columns=['Number_missing'])
missingvalue['Percentage_missing']=np.round(100*missingvalue['Number_missing']/len(data),2)
missingvalue

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
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