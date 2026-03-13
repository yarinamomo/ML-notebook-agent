# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train_df = pd.merge(train_df, greeks_df, on="Id")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# Remove the first column
train_df = train_df.drop("Id", axis=1)
test_df = test_df.drop("Id", axis=1)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# # One-hot encoding
# encoder = OneHotEncoder(handle_unknown="ignore")
# train_df = pd.get_dummies(train_df, columns=list(train_df))
# test_df = pd.get_dummies(test_df, columns=list(test_df))

# === AFTER (edited) ===
# Identify categorical columns (non-numeric columns)
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

# Apply one-hot encoding to categorical columns only
if categorical_cols:
    train_df = pd.get_dummies(train_df, columns=categorical_cols, drop_first=False)
    test_df = pd.get_dummies(test_df, columns=categorical_cols, drop_first=False)
    
    # Ensure train and test have the same columns after encoding
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    # Add missing columns to test with zeros and vice versa
    for col in train_cols - test_cols:
        test_df[col] = 0
    for col in test_cols - train_cols:
        train_df[col] = 0
    
    # Reorder test columns to match train
    test_df = test_df[train_df.columns]

print(f"Train shape after encoding: {train_df.shape}")
print(f"Test shape after encoding: {test_df.shape}")

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)