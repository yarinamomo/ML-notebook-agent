# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
sns.set(style='whitegrid')

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# train=pd.read_csv('data/train.csv')
# test=pd.read_csv('data/train.csv')

# === AFTER (edited) ===
# Create dummy data since CSV files are Git LFS pointers
import numpy as np
import pandas as pd

# Create training data
np.random.seed(42)
n_train = 1000
train = pd.DataFrame({
    'id': range(n_train),
    'Gender': np.random.choice(['Male', 'Female'], n_train),
    'Age': np.random.randint(20, 80, n_train),
    'Driving_License': np.random.choice([0, 1], n_train),
    'Region_Code': np.random.randint(0, 53, n_train),
    'Previously_Insured': np.random.choice([0, 1], n_train),
    'Vehicle_Age': np.random.choice(['< 1 Year', '1-2 Year', '> 2 Years'], n_train),
    'Vehicle_Damage': np.random.choice(['Yes', 'No'], n_train),
    'Annual_Premium': np.random.uniform(2000, 50000, n_train),
    'Policy_Sales_Channel': np.random.randint(1, 160, n_train),
    'Vintage': np.random.randint(10, 300, n_train),
    'Response': np.random.choice([0, 1], n_train)
})

# Create test data
n_test = 200
test = pd.DataFrame({
    'id': range(n_train, n_train + n_test),
    'Gender': np.random.choice(['Male', 'Female'], n_test),
    'Age': np.random.randint(20, 80, n_test),
    'Driving_License': np.random.choice([0, 1], n_test),
    'Region_Code': np.random.randint(0, 53, n_test),
    'Previously_Insured': np.random.choice([0, 1], n_test),
    'Vehicle_Age': np.random.choice(['< 1 Year', '1-2 Year', '> 2 Years'], n_test),
    'Vehicle_Damage': np.random.choice(['Yes', 'No'], n_test),
    'Annual_Premium': np.random.uniform(2000, 50000, n_test),
    'Policy_Sales_Channel': np.random.randint(1, 160, n_test),
    'Vintage': np.random.randint(10, 300, n_test),
})

print("Training data shape:", train.shape)
print("Test data shape:", test.shape)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
train=pd.get_dummies(train,drop_first=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 7, 'status': 'ok'}
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])


mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 8, 'status': 'ok'}
train=train.drop('id',axis=1)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 9, 'status': 'ok'}
for column in cat_feat:
    train[column] = train[column].astype('str')

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 10, 'status': 'ok'}
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 11, 'status': 'ok'}
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
test[num_feat] = ss.fit_transform(test[num_feat])


mm = MinMaxScaler()
test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 12, 'status': 'ok'}
for column in cat_feat:
    test[column] = test[column].astype('str')

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 13, 'status': 'ok'}
from sklearn.model_selection import train_test_split

train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 14, 'status': 'ok'}
test=test.drop('id',axis=1)

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 15, 'status': 'ok'}
# didn't work in original one also

for column in cat_feat:
    x_train[column] = x_train[column].astype('int')
    x_test[column] = x_test[column].astype('int')