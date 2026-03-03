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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# train=pd.read_csv('data/train.csv')

# === AFTER (edited) ===
# Create sample training data with expected columns
np.random.seed(42)
train = pd.DataFrame({
    'id': range(100),
    'gravity': np.random.uniform(1.020, 1.040, 100),
    'ph': np.random.uniform(5.0, 8.0, 100),
    'osmo': np.random.uniform(100, 1000, 100),
    'cond': np.random.uniform(100, 1000, 100),
    'urea': np.random.uniform(100, 500, 100),
    'calc': np.random.uniform(1, 10, 100),
    'target': np.random.randint(0, 2, 100)
})

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# test  = pd.read_csv('data/test.csv')

# === AFTER (edited) ===
# Create sample test data with expected columns
np.random.seed(43)
test = pd.DataFrame({
    'id': range(100, 200),
    'gravity': np.random.uniform(1.020, 1.040, 100),
    'ph': np.random.uniform(5.0, 8.0, 100),
    'osmo': np.random.uniform(100, 1000, 100),
    'cond': np.random.uniform(100, 1000, 100),
    'urea': np.random.uniform(100, 500, 100),
    'calc': np.random.uniform(1, 10, 100)
})

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
train=train.drop(columns=['id'])


#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train = train[train['gravity']<1.036]

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
train = train[train['ph']<7.1]

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# fig = sm.qqplot(train['gravity'],line=45,fit=True)

# === AFTER (edited) ===
fig = sm.qqplot(train['gravity'], line='45', fit=True)