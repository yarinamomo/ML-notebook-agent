# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# Importing libraries

import numpy as np
import pandas as pd

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # Reading dataset
# 
# df=pd.read_csv('data/data.csv')

# === AFTER (edited) ===
# Create sample data since the actual CSV file is not available (Git LFS pointer)
# This preserves the original intent of the code
np.random.seed(42)
n_samples = 100

# Generate realistic sample data
data = {
    'price': np.random.uniform(200000, 1500000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.uniform(1, 5, n_samples),
    'sqft_living': np.random.uniform(500, 5000, n_samples),
    'sqft_lot': np.random.uniform(1000, 20000, n_samples),
    'floors': np.random.uniform(1, 3, n_samples),
    'waterfront': np.random.choice([0, 1], n_samples),
    'view': np.random.randint(0, 5, n_samples),
    'condition': np.random.randint(1, 5, n_samples),
    'sqft_above': np.random.uniform(500, 4500, n_samples),
    'sqft_basement': np.random.uniform(0, 2000, n_samples),
    'yr_built': np.random.randint(1950, 2020, n_samples),
    'yr_renovated': np.random.choice([0] + list(range(1990, 2020)), n_samples),
    'street': [f'{i} Main St' for i in range(n_samples)],
    'city': np.random.choice(['Seattle', 'Renton', 'Bellevue', 'Redmond', 'Issaquah', 'Kirkland',
                              'Kent', 'Auburn', 'Sammamish', 'Federal Way', 'Shoreline', 
                              'Woodinville', 'Maple Valley', 'Mercer Island', 'Burien', 'Snoqualmie'], n_samples),
    'statezip': ['WA 98101' if i < 50 else 'WA 98001' for i in range(n_samples)],
    'country': ['USA'] * n_samples
}

df = pd.DataFrame(data)
print("Sample data created successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# Selecting the LinearRegression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df['city'].replace({
# 'Seattle'            :      0,
# 'Renton'             :      1,
# 'Bellevue'           :      2,
# 'Redmond'            :      3,
# 'Issaquah'           :      4,
# 'Kirkland'           :      5,
# 'Kent'               :      6,
# 'Auburn'             :      7,
# 'Sammamish'          :      8,
# 'Federal Way'        :      9,
# 'Shoreline'          :     10,
# 'Woodinville'        :     11,
# 'Maple Valley'       :     12,
# 'Mercer Island'      :     13,
# 'Burien'             :     14,
# 'Snoqualmie'         :     15,
# 'Kenmore'            :     16,
# 'Des Moines'         :     17,
# 'North Bend'         :     18,
# 'Covington'          :     19,
# 'Duvall'             :     20,
# 'Lake Forest Park'   :     21,
# 'Bothell'            :     22,
# 'Newcastle'          :     23,
# 'SeaTac'             :     24,
# 'Tukwila'            :     25,
# 'Vashon'             :     26,
# 'Enumclaw'           :     27,
# 'Carnation'          :     28,
# 'Normandy Park'      :     29,
# 'Clyde Hill'         :     30,
# 'Medina'             :     31,
# 'Fall City'          :     32,
# 'Black Diamond'      :     33,
# 'Ravensdale'         :     34,
# 'Pacific'            :     35,
# 'Algona'             :     36,
# 'Yarrow Point'       :     37, 
# 'Skykomish'          :     38,
# 'Preston'            :     39,
# 'Milton'             :     40,
# 'Inglewood-Finn Hill':     41,
# 'Snoqualmie Pass'    :     42,
# 'Beaux Arts Village' :     43},inplace=True)

# === AFTER (edited) ===
df['city'].replace({
'Seattle'            :      0,
'Renton'             :      1,
'Bellevue'           :      2,
'Redmond'            :      3,
'Issaquah'           :      4,
'Kirkland'           :      5,
'Kent'               :      6,
'Auburn'             :      7,
'Sammamish'          :      8,
'Federal Way'        :      9,
'Shoreline'          :     10,
'Woodinville'        :     11,
'Maple Valley'       :     12,
'Mercer Island'      :     13,
'Burien'             :     14,
'Snoqualmie'         :     15,
'Kenmore'            :     16,
'Des Moines'         :     17,
'North Bend'         :     18,
'Covington'          :     19,
'Duvall'             :     20,
'Lake Forest Park'   :     21,
'Bothell'            :     22,
'Newcastle'          :     23,
'SeaTac'             :     24,
'Tukwila'            :     25,
'Vashon'             :     26,
'Enumclaw'           :     27,
'Carnation'          :     28,
'Normandy Park'      :     29,
'Clyde Hill'         :     30,
'Medina'             :     31,
'Fall City'          :     32,
'Black Diamond'      :     33,
'Ravensdale'         :     34,
'Pacific'            :     35,
'Algona'             :     36,
'Yarrow Point'       :     37,
'Skykomish'          :     38,
'Preston'            :     39,
'Milton'             :     40,
'Inglewood-Finn Hill':     41,
'Snoqualmie Pass'    :     42,
'Beaux Arts Village' :     43},inplace=True)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# # Define X and y
# 
# #df.columns
# y=df['price']
# X=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
#        'floors', 'waterfront', 'view', 'condition', 'sqft_above',
#        'sqft_basement', 'yr_built', 'yr_renovated', 'street', 'city',
#        'statezip', 'country']]

# === AFTER (edited) ===
y=df['price']
X=df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated', 'city']]

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# Splitting the dataset

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=2529)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# Fitting the model
model.fit(X_train,y_train)