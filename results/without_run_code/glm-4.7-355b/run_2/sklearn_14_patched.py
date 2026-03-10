# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# Importing libraries

import numpy as np
import pandas as pd

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# Reading dataset

df=pd.read_csv('data/data.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# Selecting the LinearRegression model

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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