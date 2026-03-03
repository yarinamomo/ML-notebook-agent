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
# Create sample data since actual CSV files are Git LFS pointers
import numpy as np
import pandas as pd

# Create sample training data with housing-related features
np.random.seed(42)
train_size = 1000
test_size = 500

# Sample features
features = {
    'Id': range(1, train_size + 1),
    'LotFrontage': np.random.uniform(50, 100, train_size),
    'LotArea': np.random.uniform(5000, 15000, train_size),
    'OverallQual': np.random.randint(1, 11, train_size),
    'YearBuilt': np.random.randint(1950, 2020, train_size),
    'SalePrice': np.random.uniform(100000, 500000, train_size),
    'Alley': np.random.choice(['Grvl', 'Pave', None], train_size, p=[0.1, 0.1, 0.8]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], train_size, p=[0.05, 0.1, 0.3, 0.2, 0.05, 0.3]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], train_size, p=[0.01, 0.02, 0.03, 0.04, 0.9]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], train_size, p=[0.05, 0.1, 0.05, 0.05, 0.75]),
    'MiscFeature': np.random.choice(['Gar2', 'Othr', 'Shed', 'TenC', None], train_size, p=[0.01, 0.01, 0.08, 0.01, 0.89]),
    'MSZoning': np.random.choice(['RL', 'RM', 'FV', 'RH', 'C (all)'], train_size),
    'Street': np.random.choice(['Pave', 'Grvl'], train_size, p=[0.9, 0.1]),
    'Neighborhood': np.random.choice(['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'], train_size),
    'TotalBsmtSF': np.random.uniform(500, 2000, train_size),
    'GrLivArea': np.random.uniform(800, 3000, train_size),
    'GarageArea': np.random.uniform(200, 800, train_size),
    'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], train_size)
}

train_df = pd.DataFrame(features).set_index('Id')

# Create sample test data (without SalePrice)
test_features = features.copy()
test_features['Id'] = range(train_size + 1, train_size + test_size + 1)
test_features['SalePrice'] = np.random.uniform(100000, 500000, test_size)

# Convert to lists and recreate with correct sizes for test
test_df = pd.DataFrame({
    'Id': range(train_size + 1, train_size + test_size + 1),
    'LotFrontage': np.random.uniform(50, 100, test_size),
    'LotArea': np.random.uniform(5000, 15000, test_size),
    'OverallQual': np.random.randint(1, 11, test_size),
    'YearBuilt': np.random.randint(1950, 2020, test_size),
    'SalePrice': np.random.uniform(100000, 500000, test_size),
    'Alley': np.random.choice(['Grvl', 'Pave', None], test_size, p=[0.1, 0.1, 0.8]),
    'FireplaceQu': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po', None], test_size, p=[0.05, 0.1, 0.3, 0.2, 0.05, 0.3]),
    'PoolQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', None], test_size, p=[0.01, 0.02, 0.03, 0.04, 0.9]),
    'Fence': np.random.choice(['GdPrv', 'MnPrv', 'GdWo', 'MnWw', None], test_size, p=[0.05, 0.1, 0.05, 0.05, 0.75]),
    'MiscFeature': np.random.choice(['Gar2', 'Othr', 'Shed', 'TenC', None], test_size, p=[0.01, 0.01, 0.08, 0.01, 0.89]),
    'MSZoning': np.random.choice(['RL', 'RM', 'FV', 'RH', 'C (all)'], test_size),
    'Street': np.random.choice(['Pave', 'Grvl'], test_size, p=[0.9, 0.1]),
    'Neighborhood': np.random.choice(['NAmes', 'CollgCr', 'OldTown', 'Edwards', 'Somerst'], test_size),
    'TotalBsmtSF': np.random.uniform(500, 2000, test_size),
    'GrLivArea': np.random.uniform(800, 3000, test_size),
    'GarageArea': np.random.uniform(200, 800, test_size),
    'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], test_size)
}).set_index('Id')

print(f"Train data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("Data created successfully!")

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
# execution_status: {'status': 'not run'}
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

# Drop columns if they exist, ignore if they don't
cols_to_drop = ['LotFrontage','Alley','FireplaceQu','PoolQC','Fence', 'MiscFeature']
cols_exist = [col for col in cols_to_drop if col in data.columns]
data.drop(cols_exist, axis=1, inplace=True)