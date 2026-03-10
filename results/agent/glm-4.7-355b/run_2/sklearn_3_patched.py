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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# # One-hot encoding
# encoder = OneHotEncoder(handle_unknown="ignore")
# train_df = pd.get_dummies(train_df, columns=list(train_df))
# test_df = pd.get_dummies(test_df, columns=list(test_df))

# === AFTER (edited) ===
# Identify categorical columns to encode (all non-numeric columns)
categorical_cols = ['EJ', 'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']

# Get the combined set of columns for consistent one-hot encoding
train_cat = train_df[categorical_cols]
test_cat = test_df[['EJ']]  # EJ is the only categorical column in test

combined_df = pd.concat([train_cat, test_cat], axis=0, ignore_index=True)
combined_dummies = pd.get_dummies(combined_df, columns=categorical_cols)

# Split back into train and test with consistent columns
train_dummies = combined_dummies.iloc[:len(train_df)].reset_index(drop=True)
test_dummies = combined_dummies.iloc[len(train_df):].reset_index(drop=True)

# Remove categorical columns and add one-hot encoded columns
train_df = pd.concat([train_df.drop(categorical_cols, axis=1).reset_index(drop=True), train_dummies], axis=1)
test_df = pd.concat([test_df.drop(['EJ'], axis=1).reset_index(drop=True), test_dummies], axis=1)

# Align test_df columns with train_df columns (fill missing columns with 0)
test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)