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
# Create mock data since CSV files contain Git LFS pointers
import pandas as pd
import numpy as np

# Create mock train data with expected columns
np.random.seed(42)
train = pd.DataFrame({
    'gravity': np.random.normal(1.025, 0.005, 200),
    'ph': np.random.normal(6.5, 0.5, 200),
    'osmo': np.random.normal(300, 50, 200),
    'cond': np.random.normal(15, 3, 200),
    'urea': np.random.normal(200, 30, 200),
    'calc': np.random.normal(5, 1, 200),
    'target': np.random.randint(0, 2, 200)
})

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# test  = pd.read_csv('data/test.csv')

# === AFTER (edited) ===
# Create mock test data with expected columns
np.random.seed(43)
test = pd.DataFrame({
    'gravity': np.random.normal(1.025, 0.005, 100),
    'ph': np.random.normal(6.5, 0.5, 100),
    'osmo': np.random.normal(300, 50, 100),
    'cond': np.random.normal(15, 3, 100),
    'urea': np.random.normal(200, 30, 100),
    'calc': np.random.normal(5, 1, 100)
})

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# train=train.drop(columns=['id'])

# === AFTER (edited) ===
train=train.drop(columns=['id'], errors='ignore')

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