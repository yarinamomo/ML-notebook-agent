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
# Identify categorical columns (object type) that exist in both train and test
train_cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
test_cat_cols = test_df.select_dtypes(include=['object']).columns.tolist()

# Keep only categorical columns that exist in both datasets
common_categorical_cols = [col for col in train_cat_cols if col in test_cat_cols]

# One-hot encode only common categorical columns
if common_categorical_cols:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Get encoded features for train
    train_encoded = encoder.fit_transform(train_df[common_categorical_cols])
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(common_categorical_cols), index=train_df.index)
    
    # Get encoded features for test
    test_encoded = encoder.transform(test_df[common_categorical_cols])
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(common_categorical_cols), index=test_df.index)
    
    # Drop original categorical columns and add encoded ones
    train_df = pd.concat([train_df.drop(common_categorical_cols, axis=1), train_encoded_df], axis=1)
    test_df = pd.concat([test_df.drop(common_categorical_cols, axis=1), test_encoded_df], axis=1)

# Drop any remaining categorical columns (those only in train or only in test)
train_df = train_df.select_dtypes(exclude=['object'])
test_df = test_df.select_dtypes(exclude=['object'])

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
# Find common columns between train and test dataframes
common_cols = [col for col in train_df.columns if col in test_df.columns]

# Scale only common columns
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[common_cols])
test_scaled = scaler.transform(test_df[common_cols])

# Convert back to DataFrame with column names
train_df = pd.DataFrame(train_scaled, columns=common_cols)
test_df = pd.DataFrame(test_scaled, columns=common_cols)