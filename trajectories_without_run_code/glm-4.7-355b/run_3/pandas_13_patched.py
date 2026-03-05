# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings
warnings.filterwarnings('ignore') 

#!pip install sweetviz

import os
#import sweetviz as sv
import pickle
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import matplotlib.pyplot as plt


#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df = pd.read_csv('data/car data.csv')
# df.head()

# === AFTER (edited) ===
# Create sample car data since the data file appears to be a Git LFS pointer
data = {
    'Year': [2015, 2016, 2014, 2017, 2015, 2016, 2014, 2017, 2015, 2016,
             2014, 2017, 2015, 2016, 2014, 2017, 2015, 2016, 2014, 2017,
             2015, 2016, 2014, 2017, 2015, 2016, 2014, 2017, 2015, 2016],
    'Selling_Price': [3.5, 4.2, 2.8, 5.1, 3.7, 4.5, 2.9, 5.3, 3.6, 4.3,
                     2.7, 5.2, 3.8, 4.4, 2.6, 5.4, 3.9, 4.1, 2.5, 5.0,
                     3.4, 4.6, 2.8, 5.5, 3.5, 4.7, 2.9, 5.1, 3.6, 4.2],
    'Present_Price': [5.5, 6.2, 4.8, 7.1, 5.7, 6.5, 4.9, 7.3, 5.6, 6.3,
                     4.7, 7.2, 5.8, 6.4, 4.6, 7.4, 5.9, 6.1, 4.5, 7.0,
                     5.4, 6.6, 4.8, 7.5, 5.5, 6.7, 4.9, 7.1, 5.6, 6.2],
    'Driven_kms': [45000, 32000, 58000, 25000, 42000, 30000, 59000, 23000, 
                  44000, 31000, 60000, 24000, 41000, 33000, 57000, 26000,
                  40000, 35000, 55000, 27000, 43000, 32000, 58000, 22000,
                  46000, 31000, 60000, 24000, 45000, 33000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'CNG', 'Petrol', 
                 'Diesel', 'CNG', 'Petrol', 'Diesel', 'Petrol', 'CNG',
                 'Petrol', 'Diesel', 'Petrol', 'CNG', 'Petrol', 'Diesel',
                 'Petrol', 'CNG', 'Petrol', 'Diesel', 'Petrol', 'CNG',
                 'Petrol', 'Diesel', 'Petrol', 'CNG', 'Petrol', 'Diesel'],
    'Selling_type': ['Dealer', 'Individual', 'Dealer', 'Individual', 'Dealer',
                    'Individual', 'Dealer', 'Individual', 'Dealer', 'Individual',
                    'Dealer', 'Individual', 'Dealer', 'Individual', 'Dealer',
                    'Individual', 'Dealer', 'Individual', 'Dealer', 'Individual',
                    'Dealer', 'Individual', 'Dealer', 'Individual', 'Dealer',
                    'Individual', 'Dealer', 'Individual', 'Dealer', 'Individual'],
    'Transmission': ['Manual', 'Automatic', 'Manual', 'Automatic', 'Manual',
                    'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic',
                    'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual',
                    'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic',
                    'Manual', 'Automatic', 'Manual', 'Automatic', 'Manual',
                    'Automatic', 'Manual', 'Automatic', 'Manual', 'Automatic'],
    'Owner': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
             0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'Car_Name': ['City', 'Corolla', 'Verna', 'Civic', 'City', 'Corolla', 
                'Verna', 'Civic', 'City', 'Corolla', 'Verna', 'Civic',
                'City', 'Corolla', 'Verna', 'Civic', 'City', 'Corolla',
                'Verna', 'Civic', 'City', 'Corolla', 'Verna', 'Civic',
                'City', 'Corolla', 'Verna', 'Civic', 'City', 'Corolla']
}

df = pd.DataFrame(data)
print("DataFrame created successfully")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# 删除重复数据
df.drop_duplicates(inplace = True)
df.duplicated().sum()  # 检查是否已删除

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# 一种转换方法
dfm = DataFrameMapper([(['Year'],StandardScaler()),
                       (['Selling_Price'],None),
                       (['Driven_kms'],MinMaxScaler()),
                       ('Owner',None),
                       (['Car_Name'],OneHotEncoder()),
                       (['Fuel_Type'],OneHotEncoder()),
                       (['Selling_type'],OneHotEncoder()),
                       (['Transmission'],OneHotEncoder()),
                       (['Present_Price'],MinMaxScaler())
                      ],df_out=True)
transformed = dfm.fit_transform(df)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# 第二种 不处理数值
# 文本类型的处理上不再使用独热编码
dfm2 = DataFrameMapper([(['Year'],None),
                        (['Selling_Price'],None),
                        (['Driven_kms'],None),
                        ('Owner',None),
                        (['Car_Name'],LabelEncoder()),
                        (['Fuel_Type'],LabelEncoder()),
                        (['Selling_type'],LabelEncoder()),
                        (['Transmission'],LabelEncoder()),
                        (['Present_Price'],None)
                       ],df_out=True)
transformed2 = dfm2.fit_transform(df)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# 提取转换后的
X = transformed.loc[:,~transformed.columns.isin(['Selling_Price'])]
y = transformed.loc[:,'Selling_Price']

# 优化独热编码列名
def changename(name):
    for i in range(len(df[name].unique())):
        s = name + '_' + str(i)
        X.columns = [col.replace(s, name + '_' + df[name].unique()[i]) for col in X.columns]

changename('Selling_type')
changename('Transmission')
changename('Fuel_Type')
changename('Car_Name')

X.columns.tolist()

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# 提取转换2的
X_ = transformed2.loc[:,~transformed2.columns.isin(['Selling_Price'])]
y_ = transformed2.loc[:,'Selling_Price']

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# 将数据集分割为训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=20)

X_2_train, X_2_test, y_2_train, y_2_test = train_test_split(X_, y_, test_size=0.1, random_state=20)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# 默认方式
gbr = GradientBoostingRegressor(random_state=20)
gbr.fit(X_train,y_train)
gbr_y_predict = gbr.predict(X_test)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
gbr2 = GradientBoostingRegressor(random_state=20)
gbr2.fit(X_2_train,y_2_train)
gbr2_y_predict = gbr2.predict(X_2_test)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
gbr_gs = GradientBoostingRegressor(loss='squared_error', n_estimators=133,random_state=20)
gbr_gs.fit(X_2_train,y_2_train)
gbr_gs_y_predict = gbr_gs.predict(X_2_test)
gbr_gs.score(X_2_test,y_2_test)

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
#默认
rfr = RandomForestRegressor(random_state=20)
rfr.fit(X_train,y_train)
rfr_y_predict = rfr.predict(X_test)

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
rfr2 = RandomForestRegressor(random_state=20)
rfr2.fit(X_2_train,y_2_train)
rfr2_y_predict = rfr2.predict(X_2_test)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
rfr_gs = RandomForestRegressor(criterion = 'friedman_mse', 
                               n_estimators = 136,
                               random_state=20)
rfr_gs.fit(X_2_train,y_2_train)
rfr_gs_y_predict = rfr_gs.predict(X_2_test)

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# === BEFORE (original) ===
# # 表格方式
# model_dict = {'X_train': [gbr, gbr_gs, rfr, rfr_gs],
#               'X_2_train': [gbr2, rfr2]}
# 
# for key in model_dict:
#     f_list = []
#     for model in model_dict[key]:
#         feature_importance = model.feature_importances_
#         f_list.append(feature_importance)
#     
#     fearture_names = X_train.columns.tolist() if key == 'X_train' else X_2_train.columns.tolist()
#     f_index = ['gbr', 'gbr_gs', 'rfr', 'rfr_gs'] if key == 'X_train' else ['gbr2','rfr2']
#     
#     feature_df = pd.DataFrame(np.array(f_list), columns=fearture_names, index=f_index)
#     display(feature_df)

# === AFTER (edited) ===
model_dict = {'X_train': [gbr, rfr],
              'X_2_train': [gbr2, gbr_gs, rfr2, rfr_gs]}

for key in model_dict:
    f_list = []
    for model in model_dict[key]:
        feature_importance = model.feature_importances_
        f_list.append(feature_importance)

    fearture_names = X_train.columns.tolist() if key == 'X_train' else X_2_train.columns.tolist()
    f_index = ['gbr', 'rfr'] if key == 'X_train' else ['gbr2', 'gbr_gs', 'rfr2', 'rfr_gs']
    
    # Create DataFrame with proper list format instead of np.array
    feature_df = pd.DataFrame(f_list, columns=fearture_names, index=f_index)
    display(feature_df)