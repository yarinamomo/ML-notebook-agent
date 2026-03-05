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
# Create a synthetic house price dataset since the actual data is not available
np.random.seed(42)
n_samples = 1000

# Generate synthetic features
houseprice = pd.DataFrame({
    'price': np.random.normal(540000, 150000, n_samples),
    'date': pd.date_range('2014-05-01', periods=n_samples, freq='D'),
    'city': np.random.choice(['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Renton'], n_samples),
    'street': [f'Street_{i}' for i in range(n_samples)],
    'statezip': ['WA 981xx'] * n_samples,
    'condition': np.random.randint(1, 6, n_samples),
    'bedrooms': np.random.randint(1, 7, n_samples),
    'bathrooms': np.random.randint(1, 5, n_samples),
    'sqft_living': np.random.normal(2000, 600, n_samples),
    'sqft_lot': np.random.normal(5000, 2000, n_samples),
    'floors': np.random.randint(1, 4, n_samples),
    'waterfront': np.random.randint(0, 2, n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'sqft_above': np.random.normal(1800, 500, n_samples),
    'sqft_basement': np.random.normal(300, 200, n_samples),
    'yr_built': np.random.randint(1950, 2016, n_samples),
    'yr_renovated': np.random.choice([0, 1990, 2000, 2010, 2015], n_samples),
})

# Clean up data - ensure positive values
houseprice['price'] = houseprice['price'].clip(100000, 1500000)
houseprice['sqft_living'] = houseprice['sqft_living'].clip(500, 5000)
houseprice['sqft_lot'] = houseprice['sqft_lot'].clip(1000, 20000)

print("Synthetic house price dataset created")
print(f"Shape: {houseprice.shape}")
print(f"Columns: {houseprice.columns.tolist()}")
print("\nFirst few rows:")
print(houseprice.head())

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
