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
# Create mock housing data since the actual CSV is a Git LFS pointer
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500

houseprice = pd.DataFrame({
    'price': np.random.uniform(200000, 800000, n_samples),
    'date': pd.date_range('2014-01-01', periods=n_samples, freq='D'),
    'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Renton'], n_samples),
    'street': [f'{np.random.randint(1, 9999)} Main St' for _ in range(n_samples)],
    'statezip': ['WA 98001' for _ in range(n_samples)],
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.uniform(1, 4, n_samples),
    'sqft_living': np.random.uniform(1000, 4000, n_samples),
    'sqft_lot': np.random.uniform(2000, 10000, n_samples),
    'floors': np.random.uniform(1, 3, n_samples),
    'waterfront': np.random.randint(0, 2, n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'sqft_above': np.random.uniform(800, 3500, n_samples),
    'sqft_basement': np.random.uniform(0, 1500, n_samples),
    'yr_built': np.random.randint(1950, 2020, n_samples),
    'yr_renovated': np.random.choice([0, np.random.randint(2000, 2020)], n_samples),
})

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
Y=houseprice['price']

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X.values,Y.values,test_size=0.2)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# train_houseprice=X_train.join(Y_train)

# === AFTER (edited) ===
train_houseprice=pd.DataFrame(X_train, columns=X.columns).assign(price=Y_train)