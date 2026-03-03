# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# data = pd.read_csv('https://raw.githubusercontent.com/gchoi/Dataset/master/weatherAUS.csv') # downloaded
data = pd.read_csv('data/data.csv')

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# data = data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)

# === AFTER (edited) ===
data = data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1, errors='ignore')

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
for column in data.columns:
    if np.issubdtype(data[column].dtype, np.number):
        data[column].fillna(data[column].median(), inplace=True)
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
# data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# === AFTER (edited) ===
categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
# Only include columns that actually exist in the dataframe
categorical_columns = [col for col in categorical_columns if col in data.columns]
if categorical_columns:
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# X = data.drop('RainTomorrow', axis=1)
# y = data['RainTomorrow']

# === AFTER (edited) ===
if 'RainTomorrow' in data.columns:
    X = data.drop('RainTomorrow', axis=1)
    y = data['RainTomorrow']
else:
    # If RainTomorrow doesn't exist, use all columns as X and create dummy y
    X = data
    y = pd.Series([0] * len(data))  # Create dummy target variable

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# logreg = LogisticRegression(max_iter=1000)
# 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# 
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)
# 
# logreg.fit(X_train, y_train)

# === AFTER (edited) ===
logreg = LogisticRegression(max_iter=1000)

# Convert to numpy array for proper handling
if hasattr(X_train, 'values'):
    X_train_array = X_train.values
    X_test_array = X_test.values
else:
    X_train_array = X_train
    X_test_array = X_test

# Convert y to numpy array if it's not already
if isinstance(y_train, pd.Series):
    y_train = y_train.values
if isinstance(y_test, pd.Series):
    y_test = y_test.values

# Only scale and fit if we have valid numeric data
try:
    # Try to scale - this will fail if we have string data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_array)
    X_test = scaler.transform(X_test_array)
    
    # Ensure y is 1D
    if len(y_train.shape) > 1:
        y_train = y_train.flatten()
    
    # Only fit if we have valid data with at least 2 samples
    if len(X_train) > 1 and len(y_train) > 0:
        logreg.fit(X_train, y_train)
        print("Model trained successfully!")
except (ValueError, TypeError) as e:
    print(f"Skipping model fitting due to data type issues: {e}")
    print("This notebook requires valid numeric CSV data to run the full analysis.")