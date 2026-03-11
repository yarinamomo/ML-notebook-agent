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
# Save target variable before one-hot encoding
target = train_df['Class']
train_df = train_df.drop('Class', axis=1)

# Concatenate train and test to ensure consistent one-hot encoding
combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# One-hot encode all categorical columns: EJ, Epsilon, Alpha, Beta, Gamma, Delta
categorical_cols = ['EJ', 'Epsilon', 'Alpha', 'Beta', 'Gamma', 'Delta']
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# Split back into train and test
train_df = combined_df.iloc[:len(train_df)]
test_df = combined_df.iloc[len(train_df):]

# Add target back to train
train_df['Class'] = target.values

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
# Separate features and target from train data
target = train_df['Class']
train_features = train_df.drop('Class', axis=1)

# Save column names before scaling
feature_cols = train_features.columns
test_cols = test_df.columns

# Scale features
scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)
test_df_scaled = scaler.transform(test_df)

# Convert back to DataFrames with column names
train_df = pd.DataFrame(train_features_scaled, columns=feature_cols)
test_df = pd.DataFrame(test_df_scaled, columns=test_cols)

# Add target back to train
train_df['Class'] = target.values