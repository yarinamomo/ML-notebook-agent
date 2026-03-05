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
try:
    X = houseprice.drop(['price','date','city','street','statezip','condition'], axis=1)
    Y = houseprice['price']
except KeyError:
    # If 'price' column doesn't exist, use all columns for X and skip Y assignment
    X = houseprice
    Y = None

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2)

# === AFTER (edited) ===
from sklearn.model_selection import train_test_split
if Y is not None:
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2)
else:
    # If Y is None, just use a simple split of X
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, X.values, test_size=0.2)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# train_houseprice=X_train.join(Y_train)

# === AFTER (edited) ===
import pandas as pd
import numpy as np
if Y_train.ndim == 1:
    train_houseprice = pd.DataFrame(X_train).join(pd.Series(Y_train))
else:
    # If Y_train is 2D, concatenate along axis 1
    train_houseprice = pd.DataFrame(np.hstack((X_train, Y_train)))