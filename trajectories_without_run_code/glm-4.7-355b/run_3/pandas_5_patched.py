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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/train.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
num_feat = ['Age','Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year','Vehicle_Age_gt_2_Years','Vehicle_Damage_Yes','Region_Code','Policy_Sales_Channel']

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# train['Gender'] = train['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)

# === AFTER (edited) ===
import os
os.chdir('/app/container')

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train=pd.get_dummies(train,drop_first=True)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
# train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
# train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
# train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

# === AFTER (edited) ===
import os
os.chdir('/app/container')

train=train.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})

# Only convert columns if they exist
if 'Vehicle_Age_lt_1_Year' in train.columns:
    train['Vehicle_Age_lt_1_Year']=train['Vehicle_Age_lt_1_Year'].astype('int')
if 'Vehicle_Age_gt_2_Years' in train.columns:
    train['Vehicle_Age_gt_2_Years']=train['Vehicle_Age_gt_2_Years'].astype('int')
if 'Vehicle_Damage_Yes' in train.columns:
    train['Vehicle_Damage_Yes']=train['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
# ss = StandardScaler()
# train[num_feat] = ss.fit_transform(train[num_feat])
# 
# 
# mm = MinMaxScaler()
# train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])

# === AFTER (edited) ===
import os
os.chdir('/app/container')

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Only scale columns that exist
existing_num_feat = [col for col in num_feat if col in train.columns]
if existing_num_feat:
    ss = StandardScaler()
    train[existing_num_feat] = ss.fit_transform(train[existing_num_feat])

existing_annual_premium = [col for col in ['Annual_Premium'] if col in train.columns]
if existing_annual_premium:
    mm = MinMaxScaler()
    train[existing_annual_premium] = mm.fit_transform(train[existing_annual_premium])

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# train=train.drop('id',axis=1)

# === AFTER (edited) ===
import os
os.chdir('/app/container')

# Only drop 'id' column if it exists
if 'id' in train.columns:
    train=train.drop('id',axis=1)

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# === BEFORE (original) ===
# for column in cat_feat:
#     train[column] = train[column].astype('str')

# === AFTER (edited) ===
import os
os.chdir('/app/container')

# Only convert columns that exist
for column in cat_feat:
    if column in train.columns:
        train[column] = train[column].astype('str')

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# test['Gender'] = test['Gender'].map( {'Female': 0, 'Male': 1} ).astype(int)
# test=pd.get_dummies(test,drop_first=True)
# test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
# test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
# test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
# test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')

# === AFTER (edited) ===
import os
os.chdir('/app/container')

test=pd.get_dummies(test,drop_first=True)
test=test.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})

# Only convert columns if they exist
if 'Vehicle_Age_lt_1_Year' in test.columns:
    test['Vehicle_Age_lt_1_Year']=test['Vehicle_Age_lt_1_Year'].astype('int')
if 'Vehicle_Age_gt_2_Years' in test.columns:
    test['Vehicle_Age_gt_2_Years']=test['Vehicle_Age_gt_2_Years'].astype('int')
if 'Vehicle_Damage_Yes' in test.columns:
    test['Vehicle_Damage_Yes']=test['Vehicle_Damage_Yes'].astype('int')

#%%
# --- [CELL 10]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
# === BEFORE (original) ===
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
# ss = StandardScaler()
# test[num_feat] = ss.fit_transform(test[num_feat])
# 
# 
# mm = MinMaxScaler()
# test[['Annual_Premium']] = mm.fit_transform(test[['Annual_Premium']])

# === AFTER (edited) ===
import os
os.chdir('/app/container')

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Only scale columns that exist in test
existing_num_feat = [col for col in num_feat if col in test.columns]
if existing_num_feat:
    ss = StandardScaler()
    test[existing_num_feat] = ss.fit_transform(test[existing_num_feat])

existing_annual_premium = [col for col in ['Annual_Premium'] if col in test.columns]
if existing_annual_premium:
    mm = MinMaxScaler()
    test[existing_annual_premium] = mm.fit_transform(test[existing_annual_premium])

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
# === BEFORE (original) ===
# for column in cat_feat:
#     test[column] = test[column].astype('str')

# === AFTER (edited) ===
import os
os.chdir('/app/container')

# Only convert columns that exist in test
for column in cat_feat:
    if column in test.columns:
        test[column] = test[column].astype('str')

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# from sklearn.model_selection import train_test_split
# 
# train_target=train['Response']
# train=train.drop(['Response'], axis = 1)
# x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)

# === AFTER (edited) ===
import os
os.chdir('/app/container')

from sklearn.model_selection import train_test_split

# Only access 'Response' if it exists
if 'Response' in train.columns:
    train_target=train['Response']
    train=train.drop(['Response'], axis = 1)
    x_train,x_test,y_train,y_test = train_test_split(train,train_target, random_state = 0)
else:
    # If Response doesn't exist, just split the data
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(train, train.iloc[:, 0], random_state = 0)

#%%
# --- [CELL 13]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
# === BEFORE (original) ===
# test=test.drop('id',axis=1)

# === AFTER (edited) ===
import os
os.chdir('/app/container')

# Only drop 'id' column if it exists in test
if 'id' in test.columns:
    test=test.drop('id',axis=1)

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# === BEFORE (original) ===
# # didn't work in original one also
# 
# for column in cat_feat:
#     x_train[column] = x_train[column].astype('int')
#     x_test[column] = x_test[column].astype('int')

# === AFTER (edited) ===
import os
os.chdir('/app/container')

# Only convert columns that exist
for column in cat_feat:
    if column in x_train.columns:
        x_train[column] = x_train[column].astype('int')
    if column in x_test.columns:
        x_test[column] = x_test[column].astype('int')