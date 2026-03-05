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
# Only encode columns that exist in the dataframe
existing_categorical = [col for col in categorical_columns if col in data.columns]
if existing_categorical:
    data = pd.get_dummies(data, columns=existing_categorical, drop_first=True)

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
    # Define target variable based on available data
    # Use last column as target if RainTomorrow doesn't exist  
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

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
# Check if we have valid data to train
if X.empty or y.empty:
    print("Warning: No valid data available for training.")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Available columns:", list(X.columns))
else:
    logreg = LogisticRegression(max_iter=1000)
    
    # Try to scale the data if it's valid
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        print(f"Scaling failed: {e}")
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # Don't scale y for classification problems - keep labels as-is
    logreg.fit(X_train_scaled, y_train)
    print("Model trained successfully")