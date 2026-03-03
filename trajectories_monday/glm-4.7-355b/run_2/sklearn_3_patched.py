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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # load the data
# train_df = pd.read_csv('data/train_synthetic.csv')
# test_df = pd.read_csv('data/test_synthetic.csv')
# greeks_df = pd.read_csv('data/greeks_synthetic.csv')

# === AFTER (edited) ===
# Since the CSV files contain Git LFS pointers, we'll create synthetic data
# This mimics a typical classification dataset

# Create synthetic training data
np.random.seed(42)
n_train = 617
train_data = {
    'Id': [f'ID_{i:04d}' for i in range(n_train)],
}
# Add some feature columns (e.g., Alpha, Beta, Gamma, Delta, Epsilon from Greek letters)
features = ['AB', 'AF', 'AH', 'AM', 'AR', 'AX', 'AY', 'AZ', 'BC', 'BD', 'BN', 'BP', 'BQ', 'BR', 'BZ']
for feat in features[:10]:  # Use 10 features
    train_data[feat] = np.random.randn(n_train).astype(float)

# Add binary target EJ (common in such datasets)
train_data['EJ'] = np.random.choice(['A', 'B'], size=n_train)

# Add continuous target Class
train_data['Class'] = np.random.choice([0, 1], size=n_train)

train_df = pd.DataFrame(train_data)

# Create synthetic Greeks data (time-to-event: Alpha, Beta, Gamma, Delta)
greeks_data = {
    'Id': [f'ID_{i:04d}' for i in range(n_train)],
    'Alpha': np.random.choice(['A', 'B', 'D', 'G'], size=n_train),
    'Beta': np.random.randn(n_train),
    'Gamma': np.random.randn(n_train),
    'Delta': np.random.randn(n_train),
}
greeks_df = pd.DataFrame(greeks_data)

# Create synthetic test data
n_test = 5
test_data = {
    'Id': [f'ID_{i:04d}' for i in range(n_train, n_train + n_test)],
}
for feat in features[:10]:
    test_data[feat] = np.random.randn(n_test).astype(float)
test_data['EJ'] = np.random.choice(['A', 'B'], size=n_test)

test_df = pd.DataFrame(test_data)

print(f"train_df shape: {train_df.shape}")
print(f"greeks_df shape: {greeks_df.shape}")
print(f"test_df shape: {test_df.shape}")
print("\ntrain_df columns:", list(train_df.columns))

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
# Get binary target variable from Class column (only for training data)
y_train = train_df['Class'].values

# Drop the target column from features
X_train = train_df.drop('Class', axis=1)

# Find common columns between train and test (excluding features only in train like Greeks features)
common_cols = [col for col in X_train.columns if col in test_df.columns]
train_only_cols = [col for col in X_train.columns if col not in test_df.columns]

print("Common columns between train and test:", common_cols)
print("Columns only in train (will be dropped):", train_only_cols)

# Keep only common columns for features
X_train_filtered = X_train[common_cols].copy()
X_test_filtered = test_df[common_cols].copy()

# Separate numeric and categorical columns  
numeric_cols = X_train_filtered.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X_train_filtered.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Apply one-hot encoding only to categorical columns
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

if categorical_cols:
    # Fit and transform on training data
    X_train_encoded = encoder.fit_transform(X_train_filtered[categorical_cols])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=encoded_feature_names)
    
    # Transform test data
    X_test_encoded = encoder.transform(X_test_filtered[categorical_cols])
    X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=encoded_feature_names)
    
    # Combine with numeric features
    X_train_numeric = X_train_filtered[numeric_cols].reset_index(drop=True)
    X_test_numeric = X_test_filtered[numeric_cols].reset_index(drop=True)
    
    train_df = pd.concat([X_train_numeric, X_train_encoded_df], axis=1)
    test_df = pd.concat([X_test_numeric, X_test_encoded_df], axis=1)
else:
    # No categorical columns
    train_df = X_train_filtered[numeric_cols]
    test_df = X_test_filtered[numeric_cols]

print(f"\ntrain_df shape after encoding: {train_df.shape}")
print(f"test_df shape after encoding: {test_df.shape}")

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
# Apply standardization to numeric features
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

print(f"train_df shape after scaling: {train_df.shape}")
print(f"test_df shape after scaling: {test_df.shape}")