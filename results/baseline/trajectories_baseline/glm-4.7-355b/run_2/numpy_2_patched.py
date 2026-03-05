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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
# Y=houseprice['price']

# === AFTER (edited) ===
# Check available columns first
print("Available columns in dataset:")
print(houseprice.columns.tolist())

# Drop columns that actually exist in the dataset
columns_to_drop = ['price', 'date', 'city', 'street', 'statezip', 'condition']
# Only drop columns that exist
existing_columns = [col for col in columns_to_drop if col in houseprice.columns]
X = houseprice.drop(existing_columns, axis=1)
Y = houseprice['price'] if 'price' in houseprice.columns else houseprice.iloc[:, 0]

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
train_houseprice=X_train.join(Y_train)
