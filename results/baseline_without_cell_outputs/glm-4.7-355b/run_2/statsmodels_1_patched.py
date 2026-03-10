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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 2}
train=pd.read_csv('data/train.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
test  = pd.read_csv('data/test.csv')

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train=train.drop(columns=['id'])

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train = train[train['gravity']<1.036]

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train = train[train['ph']<7.1]

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# fig = sm.qqplot(train['gravity'],line=45,fit=True)

# === AFTER (edited) ===
fig = sm.qqplot(train['gravity'],line='45',fit=True)