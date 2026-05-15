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
# Identify categorical and numeric columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Remove 'Class' from numeric_cols as it's the target variable and not in test_df
if 'Class' in numeric_cols:
    numeric_cols.remove('Class')

print("Train categorical columns:", categorical_cols)
print("Test categorical columns:", test_df.select_dtypes(include=['object']).columns.tolist())
print("\nNumeric columns (excluding Class):", numeric_cols)

# Categorical columns that exist in both datasets
common_categorical_cols = [col for col in categorical_cols if col in test_df.columns]
train_only_categorical_cols = [col for col in categorical_cols if col not in test_df.columns]

print("\nCommon categorical columns:", common_categorical_cols)
print("Train-only categorical columns:", train_only_categorical_cols)

# Encode common categorical columns using OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_encoded = encoder.fit_transform(train_df[common_categorical_cols])
test_encoded = encoder.transform(test_df[common_categorical_cols])

# Get feature names for encoded columns
encoded_feature_names = encoder.get_feature_names_out(common_categorical_cols)

# Create DataFrames with encoded features
train_encoded_df = pd.DataFrame(train_encoded, columns=encoded_feature_names, index=train_df.index)
test_encoded_df = pd.DataFrame(test_encoded, columns=encoded_feature_names, index=test_df.index)

# For train-only categorical columns, encode them separately
if train_only_categorical_cols:
    train_only_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    train_only_encoded = train_only_encoder.fit_transform(train_df[train_only_categorical_cols])
    train_only_encoded_df = pd.DataFrame(train_only_encoded, 
                                        columns=train_only_encoder.get_feature_names_out(train_only_categorical_cols),
                                        index=train_df.index)
else:
    train_only_encoded_df = pd.DataFrame(index=train_df.index)

# Create empty DataFrame for test with train-only encoded columns (all zeros)
test_only_encoded_df = pd.DataFrame(0, index=test_df.index, columns=train_only_encoded_df.columns)

# Ensure numeric columns exist in both datasets
common_numeric_cols = [col for col in numeric_cols if col in test_df.columns]

# Combine everything
train_df = pd.concat([train_df[numeric_cols], train_encoded_df, train_only_encoded_df], axis=1)
test_df = pd.concat([test_df[common_numeric_cols], test_encoded_df, test_only_encoded_df], axis=1)

print("\nTrain shape after encoding:", train_df.shape)
print("Test shape after encoding:", test_df.shape)
print("\nTrain column count:", len(train_df.columns))
print("Test column count:", len(test_df.columns))

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)