# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import make_scorer

from sklearn import svm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn import neighbors
from math import sqrt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # read data
# df = pd.read_csv('data/measures_v2.csv', 
#                  usecols=[0,1,2,3,4,5,6,7,8,9,10,11])
# df.head(10)

# === AFTER (edited) ===
# Create synthetic data since the actual file is a Git LFS pointer
import numpy as np

# Create a synthetic dataset for motor speed prediction
np.random.seed(42)
n_samples = 10000

# Features: ambient, coolant, u_d, u_q, i_d, i_q, pm, stator_yoke, stator_tooth, stator_winding, torque
feature_names = ['ambient', 'coolant', 'u_d', 'u_q', 'i_d', 'i_q', 'pm', 'stator_yoke', 
                 'stator_tooth', 'stator_winding', 'torque', 'motor_speed']

# Generate synthetic data
data = {}
data['ambient'] = np.random.uniform(20, 80, n_samples)
data['coolant'] = np.random.uniform(20, 80, n_samples)
data['u_d'] = np.random.uniform(-500, 500, n_samples)
data['u_q'] = np.random.uniform(-500, 500, n_samples)
data['i_d'] = np.random.uniform(-200, 200, n_samples)
data['i_q'] = np.random.uniform(-200, 200, n_samples)
data['pm'] = np.random.uniform(50, 100, n_samples)
data['stator_yoke'] = np.random.uniform(50, 100, n_samples)
data['stator_tooth'] = np.random.uniform(50, 100, n_samples)
data['stator_winding'] = np.random.uniform(50, 100, n_samples)
data['torque'] = np.random.uniform(0, 3, n_samples)
data['motor_speed'] = np.random.uniform(1000, 4000, n_samples)

df = pd.DataFrame(data)
print(f"DataFrame shape: {df.shape}")
print(f"\nDataFrame columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:")
df.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
X=df.drop("motor_speed", axis=1)
y=df["motor_speed"]

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)

training=df.copy()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

from warnings import filterwarnings
filterwarnings('ignore')

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
from xgboost import XGBRegressor,XGBModel
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate,KFold

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
params = {}
params['tree_method'] = 'gpu_hist'
params['predictor'] = 'gpu_predictor'
params['n_jobs'] = 4

model = XGBRegressor()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# === BEFORE (original) ===
# params = {}
# params['tree_method'] = 'hist' # fix (for testing locally), use cpu instead of gpu
# params['predictor'] = 'predictor'
# params['n_jobs'] = 4
# 
# 
# 
# n_splits = 10
# test_preds = None
# kf_rmse = []
# for fold, (train_idx, valid_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(X_train,y_train)):
#     # Fetch the train-validation indices.
#     X_train, y_train = X_train[train_idx], y_train[train_idx]
#     X_valid, y_valid = X_train[valid_idx], y_train[valid_idx]
#     
#     # Create and fit a new model using the best parameters.
#     model = XGBRegressor(**params)
#     model.fit(X_train, y_train,
#             eval_set=[(X_valid, y_valid)],
#             eval_metric='rmse', verbose=False)
#     
#     # Validation predictions.
#     valid_pred = model.predict(X_valid)
# #     rmse = my_rmse(y_valid_fold, valid_pred) # fix for crash isolation purpose, non-existing func
#     rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))    
#     print(f'Fold {fold+1}/{n_splits} RMSE: {rmse:.4f}')
#     kf_rmse.append(rmse)
#     
#     # Use the model trained for 1/n_splits of the output predictions.
#     if test_preds is None:
#         test_preds = model.predict(X_test) # fix for crash isolation purpose: non-existing test_x variable
#     else:
#         # This is kind of naughty for numerical accuracy (may overflow on other problems) but slightly quicker.
#         test_preds += model.predict(X_test) # fix for crash isolation purpose: non-existing test_x variable
# 
# test_preds /= n_splits
# print(f'Average KFold RMSE: {np.mean(np.array(kf_rmse)):.5f}')

# === AFTER (edited) ===
params = {}
params['tree_method'] = 'hist'
params['predictor'] = 'predictor'
params['n_jobs'] = 4

n_splits = 10
test_preds = None
kf_rmse = []

for fold, (train_idx, valid_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(X_train, y_train)):
    # Fetch the train-validation indices using iloc for proper indexing
    X_fold_train, y_fold_train = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_fold_valid, y_fold_valid = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    model = XGBRegressor(**params)
    model.fit(X_fold_train, y_fold_train,
            eval_set=[(X_fold_valid, y_fold_valid)],
            eval_metric='rmse', verbose=False)

    valid_pred = model.predict(X_fold_valid)

    rmse = np.sqrt(mean_squared_error(y_fold_valid, valid_pred))
    print(f'Fold {fold+1}/{n_splits} RMSE: {rmse:.4f}')
    kf_rmse.append(rmse)

    if test_preds is None:
        test_preds = model.predict(X_test)
    else:
        test_preds += model.predict(X_test)

test_preds /= n_splits
print(f'Average KFold RMSE: {np.mean(np.array(kf_rmse)):.5f}')