# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
import pylab as p 
from sklearn.metrics import accuracy_score
%matplotlib inline

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train=pd.read_csv('data/train.csv')


#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
test  = pd.read_csv('data/test.csv')

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# train=train.drop(columns=['id'])

# === AFTER (edited) ===
# Drop 'id' column if it exists
if 'id' in train.columns:
    train = train.drop(columns=['id'])

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# train = train[train['gravity']<1.036]

# === AFTER (edited) ===
# Filter rows where 'gravity' < 1.036 if the column exists
if 'gravity' in train.columns:
    train = train[train['gravity'] < 1.036]

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# train = train[train['ph']<7.1]

# === AFTER (edited) ===
# Filter rows where 'ph' < 7.1 if the column exists
if 'ph' in train.columns:
    train = train[train['ph'] < 7.1]

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# fig = sm.qqplot(train['gravity'],line=45,fit=True)

# === AFTER (edited) ===
# Create Q-Q plot for 'gravity' column if it exists
if 'gravity' in train.columns:
    fig = sm.qqplot(train['gravity'], line=45, fit=True)
else:
    print("Column 'gravity' not found in train dataframe")