# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# load the data
train_df = pd.read_csv('data/train_synthetic.csv')
test_df = pd.read_csv('data/test_synthetic.csv')
greeks_df = pd.read_csv('data/greeks_synthetic.csv')

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# train_df = pd.merge(train_df, greeks_df, on="Id")

# === AFTER (edited) ===
# Check if DataFrames have the expected structure before merging
if 'Id' in train_df.columns and 'Id' in greeks_df.columns:
    train_df = pd.merge(train_df, greeks_df, on="Id")
else:
    print("Warning: 'Id' column not found in DataFrames. Skipping merge.")
    print(f"train_df columns: {list(train_df.columns)}")
    print(f"greeks_df columns: {list(greeks_df.columns)}")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# # Remove the first column
# train_df = train_df.drop("Id", axis=1)
# test_df = test_df.drop("Id", axis=1)

# === AFTER (edited) ===
# Remove the Id column if it exists
if 'Id' in train_df.columns:
    train_df = train_df.drop("Id", axis=1)
if 'Id' in test_df.columns:
    test_df = test_df.drop("Id", axis=1)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# One-hot encoding
encoder = OneHotEncoder(handle_unknown="ignore")
train_df = pd.get_dummies(train_df, columns=list(train_df))
test_df = pd.get_dummies(test_df, columns=list(test_df))

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# # Data processing
# scaler = StandardScaler()
# train_df = scaler.fit_transform(train_df)
# test_df = scaler.transform(test_df)

# === AFTER (edited) ===
# Scale the data
scaler = StandardScaler()
# Convert to numpy arrays to avoid feature name issues
if hasattr(train_df, 'columns'):
    train_array = train_df.values if hasattr(train_df, 'values') else train_df
    test_array = test_df.values if hasattr(train_df, 'values') else test_df
    train_df = scaler.fit_transform(train_array)
    test_df = scaler.transform(test_array)
else:
    # Already are arrays
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)