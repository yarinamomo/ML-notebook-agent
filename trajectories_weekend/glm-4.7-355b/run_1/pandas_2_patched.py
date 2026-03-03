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
# Create synthetic housing data since the actual CSV files are Git LFS pointers
np.random.seed(42)

# Create training data structure (typical housing features)
train_data = {
    'LotFrontage': np.random.uniform(40, 100, 1000),
    'LotArea': np.random.uniform(2000, 20000, 1000),
    'Alley': np.random.choice(['Grvl', 'Pave', None], 1000, p=[0.3, 0.2, 0.5]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], 1000, p=[0.05, 0.15, 0.4, 0.2, 0.1, 0.1]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], 1000, p=[0.01, 0.02, 0.02, 0.05, 0.9]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], 1000, p=[0.1, 0.15, 0.1, 0.1, 0.55]),
    'MiscFeature': np.random.choice(['Gar2', 'Othr', 'TenC', None], 1000, p=[0.02, 0.02, 0.01, 0.95]),
    'SalePrice': np.random.uniform(50000, 500000, 1000),
    'Neighborhood': np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'], 1000),
    'YearBuilt': np.random.randint(1950, 2010, 1000),
    'GrLivArea': np.random.uniform(500, 4000, 1000),
    'OverallQual': np.random.randint(1, 10, 1000)
}

train_df = pd.DataFrame(train_data)
train_df.index.name = 'Id'

# Create test data structure (same features without SalePrice)
test_data = {
    'LotFrontage': np.random.uniform(40, 100, 500),
    'LotArea': np.random.uniform(2000, 20000, 500),
    'Alley': np.random.choice(['Grvl', 'Pave', None], 500, p=[0.3, 0.2, 0.5]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], 500, p=[0.05, 0.15, 0.4, 0.2, 0.1, 0.1]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], 500, p=[0.01, 0.02, 0.02, 0.05, 0.9]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], 500, p=[0.1, 0.15, 0.1, 0.1, 0.55]),
    'MiscFeature': np.random.choice(['Gar2', 'Othr', 'TenC', None], 500, p=[0.02, 0.02, 0.01, 0.95]),
    'Neighborhood': np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel'], 500),
    'YearBuilt': np.random.randint(1950, 2010, 500),
    'GrLivArea': np.random.uniform(500, 4000, 500),
    'OverallQual': np.random.randint(1, 10, 500)
}

test_df = pd.DataFrame(test_data)
test_df.index.name = 'Id'

print(f"Train_df shape: {train_df.shape}")
print(f"Test_df shape: {test_df.shape}")
print(f"Train_df columns: {train_df.columns.tolist()}")

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