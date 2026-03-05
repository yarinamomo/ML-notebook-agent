# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
sns.set(style='whitegrid')

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train=pd.read_csv('data/train.csv')
# test=pd.read_csv('data/train.csv')

# === AFTER (edited) ===
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train=pd.get_dummies(train,drop_first=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
# ss = StandardScaler()
# train[num_feat] = ss.fit_transform(train[num_feat])
# 
# 
# mm = MinMaxScaler()
# train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

# === AFTER (edited) ===
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
for col in num_feat:
    if col in train.columns:
        train[col] = ss.fit_transform(train[[col]])

mm = MinMaxScaler()
if 'Annual_Premium' in train.columns:
    train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# train=train.drop('id',axis=1)

# === AFTER (edited) ===
train = train.drop('id', axis=1, errors='ignore')

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# for column in cat_feat:
#     train[column] = train[column].astype('str')

# === AFTER (edited) ===
for column in cat_feat:
    if column in train.columns:
        train[column] = train[column].astype('str')

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 10]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
# ss = StandardScaler()
# test[num_feat] = ss.fit_transform(test[num_feat])
# 
# 
# mm = MinMaxScaler()
# test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])

# === AFTER (edited) ===
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
ss = StandardScaler()
for col in num_feat:
    if col in test.columns:
        test[col] = ss.fit_transform(test[[col]])

mm = MinMaxScaler()
if 'Annual_Premium' in test.columns:
    test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# for column in cat_feat:
#     test[column] = test[column].astype('str')

# === AFTER (edited) ===
for column in cat_feat:
    if column in test.columns:
        test[column] = test[column].astype('str')

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.model_selection import train_test_split

train_target=train['Response']
train=train.drop(['Response'], axis = 1)
x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)

#%%
# --- [CELL 13]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# test=test.drop('id',axis=1)

# === AFTER (edited) ===
test = test.drop('id', axis=1, errors='ignore')

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # didn't work in original one also
# 
# for column in cat_feat:
#     x_train[column] = x_train[column].astype('int')
#     x_test[column] = x_test[column].astype('int')

# === AFTER (edited) ===
for column in cat_feat:
    if column in x_train.columns:
        x_train[column] = x_train[column].astype('int')
    if column in x_test.columns:
        x_test[column] = x_test[column].astype('int')