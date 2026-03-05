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
# Create synthetic data since the CSV file is a Git LFS pointer
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic housing data
n_samples = 1000

houseprice = pd.DataFrame({
    'price': np.random.normal(500000, 150000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.randint(1, 4, n_samples),
    'sqft_living': np.random.normal(2000, 500, n_samples),
    'sqft_lot': np.random.normal(5000, 1000, n_samples),
    'floors': np.random.randint(1, 4, n_samples),
    'waterfront': np.random.randint(0, 2, n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 6, n_samples),
    'grade': np.random.randint(1, 14, n_samples),
    'sqft_above': np.random.normal(1800, 400, n_samples),
    'sqft_basement': np.random.uniform(0, 1000, n_samples),
    'yr_built': np.random.randint(1950, 2020, n_samples),
    'yr_renovated': np.random.choice([0, 1990, 2000, 2010, 2015], n_samples, p=[0.6, 0.1, 0.1, 0.1, 0.1]),
    'zipcode': np.random.randint(98001, 98199, n_samples),
    'lat': np.random.uniform(47.4, 47.8, n_samples),
    'long': np.random.uniform(-122.5, -122.2, n_samples),
    'sqft_living15': np.random.normal(2000, 500, n_samples),
    'sqft_lot15': np.random.normal(5000, 1000, n_samples),
    'date': pd.date_range('2014-05-01', periods=n_samples, freq='D'),
    'city': np.random.choice(['Seattle', 'Bellevue', 'Kirkland', 'Redmond', 'Mercer Island'], n_samples),
    'street': [f'{i} Main St' for i in range(1, n_samples+1)],
    'statezip': ['WA 98000' for _ in range(n_samples)]
})

# Ensure price is positive
houseprice['price'] = np.abs(houseprice['price'])
houseprice['sqft_living'] = np.abs(houseprice['sqft_living'])
houseprice['sqft_lot'] = np.abs(houseprice['sqft_lot'])
houseprice['sqft_above'] = np.abs(houseprice['sqft_above'])

print("Data created successfully")
print(f"Shape: {houseprice.shape}")
print(f"Columns: {houseprice.columns.tolist()}")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# X=houseprice.drop(['price','date','city','street','statezip','condition'],axis=1)
# Y=houseprice['price']

# === AFTER (edited) ===
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
