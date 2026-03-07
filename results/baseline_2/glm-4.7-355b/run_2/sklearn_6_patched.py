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

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train_ds = pd.read_csv("data/train.csv")
test_ds = pd.read_csv("data/test.csv")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
train_ds.drop(['Id', 'MoSold', 'GarageYrBlt', 'Condition1', 'Condition2'], axis = 1, inplace = True)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
for column in train_ds:
    null_count = train_ds[column].isnull().sum()
    if null_count > 1:
        print(f"Dropping column {column} with {null_count} missing values.")
        train_ds.drop(column, axis = 1, inplace = True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
le = LabelEncoder()
string_columns = train_ds.select_dtypes(include = ['object']).columns
for column in string_columns:
    train_ds[column] = le.fit_transform(train_ds[column])

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
X = train_ds.drop(['SalePrice'], axis = 1)
y = train_ds['SalePrice']

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
from sklearn.ensemble import RandomForestRegressor
FReg = RandomForestRegressor(n_estimators = 100, random_state = 42)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
FReg.fit(X_train, y_train)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
y_pred = FReg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'R2 Score: {r2}')
print(f'MSE: {mse}')

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
test_ds_ids = test_ds['Id'] # fix for crash isolation purpose
test_ds.drop(['Id', 'MoSold', 'GarageYrBlt', 'Condition1', 'Condition2'], axis = 1, inplace = True)

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# for column in test_ds:
#     null_count = test_ds[column].isnull().sum()
#     if null_count > 1:
#         print(f"Dropping column {column} with {null_count} missing values.")
#         test_ds.drop(column, axis = 1, inplace = True)

# === AFTER (edited) ===
# Only drop columns from test_ds that were also dropped from train_ds (i.e., are not in train_ds columns)
# This ensures both datasets have the same feature set
for column in test_ds.columns:
    if column not in train_ds.columns:
        print(f"Dropping column {column} (missing in train_ds).")
        test_ds.drop(column, axis = 1, inplace = True)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
le = LabelEncoder()
string_columns = test_ds.select_dtypes(include = ['object']).columns
for column in string_columns:
    test_ds[column] = le.fit_transform(test_ds[column])

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 15}
predictions = FReg.predict(test_ds)
submissions_df = pd.DataFrame({
    "ID" : test_ds_ids, # test_data['ID'], # fix for crash isolation purpose
    "Predictions" : predictions
})

# submissions_df.to_csv('submission_csv', index = False)