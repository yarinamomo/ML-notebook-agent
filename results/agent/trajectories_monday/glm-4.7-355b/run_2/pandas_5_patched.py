# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
sns.set(style='whitegrid')

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# train=pd.read_csv('data/train.csv')
# test=pd.read_csv('data/train.csv')

# === AFTER (edited) ===
# Create sample data matching the expected structure
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

train = pd.DataFrame({
    'id': range(n_samples),
    'Gender': np.random.choice(['Female', 'Male'], n_samples),
    'Age': np.random.randint(20, 80, n_samples),
    'Driving_License': np.random.choice([0, 1], n_samples, p=[0.01, 0.99]),
    'Region_Code': np.random.randint(0, 52, n_samples),
    'Previously_Insured': np.random.choice([0, 1], n_samples),
    'Vehicle_Age': np.random.choice(['< 1 Year', '1-2 Year', '> 2 Years'], n_samples),
    'Vehicle_Damage': np.random.choice(['Yes', 'No'], n_samples),
    'Annual_Premium': np.random.uniform(2000, 80000, n_samples),
    'Policy_Sales_Channel': np.random.randint(1, 161, n_samples),
    'Vintage': np.random.randint(10, 300, n_samples),
    'Response': np.random.choice([0, 1], n_samples, p=[0.87, 0.13])
})

test = pd.DataFrame({
    'id': range(n_samples, n_samples + 500),
    'Gender': np.random.choice(['Female', 'Male'], 500),
    'Age': np.random.randint(20, 80, 500),
    'Driving_License': np.random.choice([0, 1], 500, p=[0.01, 0.99]),
    'Region_Code': np.random.randint(0, 52, 500),
    'Previously_Insured': np.random.choice([0, 1], 500),
    'Vehicle_Age': np.random.choice(['< 1 Year', '1-2 Year', '> 2 Years'], 500),
    'Vehicle_Damage': np.random.choice(['Yes', 'No'], 500),
    'Annual_Premium': np.random.uniform(2000, 80000, 500),
    'Policy_Sales_Channel': np.random.randint(1, 161, 500),
    'Vintage': np.random.randint(10, 300, 500)
})

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train=pd.get_dummies(train,drop_first=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])


mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
train=train.drop('id',axis=1)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
for column in cat_feat:
    train[column] = train[column].astype('str')

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
test[num_feat] = ss.fit_transform(test[num_feat])


mm = MinMaxScaler()
test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
for column in cat_feat:
    test[column] = test[column].astype('str')

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
from sklearn.model_selection import train_test_split

train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
test=test.drop('id',axis=1)

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# didn't work in original one also

for column in cat_feat:
    x_train[column] = x_train[column].astype('int')
    x_test[column] = x_test[column].astype('int')