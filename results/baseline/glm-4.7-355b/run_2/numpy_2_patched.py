# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
houseprice=pd.read_csv('data/data.csv')

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
# Y=houseprice['price']

# === AFTER (edited) ===
# Print available columns to identify correct names
print("Available columns:", houseprice.columns.tolist())

# Common column naming variations - check which ones exist
# Try more standard naming conventions with underscores
X=houseprice.drop(['price','date','city','street','state_zip','zip_code','house_condition','condition'], axis=1, errors='ignore')
Y=houseprice['price']

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train_houseprice=X_train.join(Y_train)
