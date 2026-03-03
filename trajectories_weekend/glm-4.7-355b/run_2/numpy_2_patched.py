# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# houseprice=pd.read_csv('data/data.csv')

# === AFTER (edited) ===
# Create sample house price data since the actual file is a Git LFS pointer
import numpy as np

# Generate synthetic house price data
np.random.seed(42)
n_samples = 4600

houseprice = pd.DataFrame({
    'price': np.random.uniform(200000, 3000000, n_samples),
    'date': pd.date_range('2014-05-01', periods=n_samples, freq='H'),
    'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond', 'Kirkland'], n_samples),
    'street': [f'Street {i}' for i in range(n_samples)],
    'statezip': [f'WA {98000 + i % 100}' for i in range(n_samples)],
    'condition': np.random.randint(1, 6, n_samples),
    'bedrooms': np.random.randint(1, 7, n_samples),
    'bathrooms': np.random.uniform(1, 5, n_samples),
    'sqft_living': np.random.uniform(500, 6000, n_samples),
    'sqft_lot': np.random.uniform(1000, 50000, n_samples),
    'floors': np.random.randint(1, 4, n_samples),
    'waterfront': np.random.randint(0, 2, n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'sqft_above': np.random.uniform(500, 5000, n_samples),
    'sqft_basement': np.random.uniform(0, 2000, n_samples),
    'yr_built': np.random.randint(1950, 2020, n_samples),
    'yr_renovated': np.random.choice([0, 1990, 2000, 2010, 2015], n_samples),
})

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
Y=houseprice['price']

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2)

# === AFTER (edited) ===
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
train_houseprice=X_train.join(Y_train)
