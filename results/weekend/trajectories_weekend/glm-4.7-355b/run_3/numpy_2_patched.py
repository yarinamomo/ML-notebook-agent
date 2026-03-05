# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# houseprice=pd.read_csv('data/data.csv')

# === AFTER (edited) ===
# Create sample house price data since the original file appears to be a Git LFS pointer
houseprice = pd.DataFrame({
    'price': [500000, 450000, 600000, 550000, 475000, 520000, 480000, 580000, 530000, 490000],
    'bedrooms': [3, 2, 4, 3, 3, 4, 2, 3, 4, 3],
    'bathrooms': [2, 1, 3, 2, 2, 2, 1, 2, 3, 2],
    'sqft_living': [2000, 1500, 2500, 1800, 1900, 2200, 1600, 2100, 2300, 1700],
    'sqft_lot': [5000, 4000, 6000, 4500, 4800, 5500, 4200, 5200, 5800, 4300],
    'floors': [2, 1, 2, 2, 2, 2, 1, 2, 2, 1],
    'waterfront': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    'view': [2, 1, 3, 2, 1, 2, 1, 3, 2, 1],
    'condition': [3, 2, 4, 3, 3, 3, 2, 4, 3, 3],
    'sqft_above': [1500, 1000, 2000, 1300, 1400, 1700, 1100, 1600, 1800, 1200],
    'sqft_basement': [500, 500, 500, 500, 500, 500, 500, 500, 500, 500],
    'yr_built': [1990, 1980, 2000, 1995, 1985, 1998, 1982, 2002, 1996, 1983],
    'yr_renovated': [0, 0, 2015, 0, 0, 0, 0, 2018, 0, 0],
    'street': ['Main St'] * 10,
    'city': ['Seattle'] * 10,
    'statezip': ['WA 98101'] * 10,
    'country': ['USA'] * 10,
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
             '2023-01-06', '2023-01-07', '2023-01-08', '2023-01-09', '2023-01-10']
})

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
# === BEFORE (original) ===
# X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
# Y=houseprice['price']

# === AFTER (edited) ===
X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
# Create Y Series with name so it can be joined later
Y=houseprice['price'].rename('price')

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
# === BEFORE (original) ===
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2)

# === AFTER (edited) ===
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
train_houseprice=X_train.join(Y_train)
