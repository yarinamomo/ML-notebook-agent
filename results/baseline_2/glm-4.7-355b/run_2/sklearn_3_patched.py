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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
# load the data
train_df = pd.read_csv('data/train_synthetic.csv')
test_df = pd.read_csv('data/test_synthetic.csv')
greeks_df = pd.read_csv('data/greeks_synthetic.csv')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train_df = pd.merge(train_df, greeks_df, on="Id")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# Remove the first column
train_df = train_df.drop("Id", axis=1)
test_df = test_df.drop("Id", axis=1)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # One-hot encoding
# encoder = OneHotEncoder(handle_unknown="ignore")
# train_df = pd.get_dummies(train_df, columns=list(train_df))
# test_df = pd.get_dummies(test_df, columns=list(test_df))

# === AFTER (edited) ===
# Identify categorical columns (non-numeric columns)
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = [col for col in train_df.columns if col not in categorical_cols and col != 'Class']

# Use OneHotEncoder for consistent encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

# Fit and transform on train data
if categorical_cols:
    train_encoded = encoder.fit_transform(train_df[categorical_cols])
    test_encoded = encoder.transform(test_df[categorical_cols])
    
    # Get feature names from encoder
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    # Create dataframes with encoded features
    train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_feature_names, index=train_df.index)
    test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_feature_names, index=test_df.index)
    
    # Combine numeric and encoded features
    train_numeric_df = train_df[numeric_cols]
    test_numeric_df = test_df[numeric_cols]
    
    train_df = pd.concat([train_numeric_df, train_encoded_df], axis=1)
    test_df = pd.concat([test_numeric_df, test_encoded_df], axis=1)
else:
    # No categorical columns, keep as is
    pass

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)