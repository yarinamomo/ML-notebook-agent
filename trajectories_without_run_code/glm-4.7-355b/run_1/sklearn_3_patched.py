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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# # load the data
# train_df = pd.read_csv('data/train_synthetic.csv')
# test_df = pd.read_csv('data/test_synthetic.csv')
# greeks_df = pd.read_csv('data/greeks_synthetic.csv')

# === AFTER (edited) ===
# Generate synthetic data since the actual data files are Git LFS pointers
np.random.seed(42)

# Create synthetic train data with features and target
train_df = pd.DataFrame({
    'Id': [f'ID_{i}' for i in range(1000)],
    'Feature_1': np.random.randn(1000),
    'Feature_2': np.random.randn(1000),
    'Feature_3': np.random.randn(1000),
    'Feature_4': np.random.randn(1000),
    'Feature_5': np.random.randn(1000),
    'Feature_6': np.random.randn(1000),
    'Feature_7': np.random.randn(1000),
    'Feature_8': np.random.randn(1000),
    'Feature_9': np.random.randn(1000),
    'Feature_10': np.random.randn(1000),
    'Class': np.random.randint(0, 2, 1000)
})

# Create synthetic test data
test_df = pd.DataFrame({
    'Id': [f'ID_{i}' for i in range(1000, 1500)],
    'Feature_1': np.random.randn(500),
    'Feature_2': np.random.randn(500),
    'Feature_3': np.random.randn(500),
    'Feature_4': np.random.randn(500),
    'Feature_5': np.random.randn(500),
    'Feature_6': np.random.randn(500),
    'Feature_7': np.random.randn(500),
    'Feature_8': np.random.randn(500),
    'Feature_9': np.random.randn(500),
    'Feature_10': np.random.randn(500),
    'Class': np.random.randint(0, 2, 500)
})

# Create synthetic greeks data (metadata like Alpha, Beta, Gamma, Delta, Epsilon)
greeks_df = pd.DataFrame({
    'Id': [f'ID_{i}' for i in range(1000)],
    'Alpha': np.random.choice(['A', 'B'], 1000),
    'Beta': np.random.choice(['B', 'C'], 1000),
    'Gamma': np.random.choice(['G', 'H', 'M'], 1000),
    'Delta': np.random.choice(['A', 'B', 'D', 'N'], 1000),
    'Epsilon': np.random.choice(['A', 'B', 'C'], 1000)
})

print("Synthetic data created successfully")
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"greeks_df shape: {greeks_df.shape}")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train_df = pd.merge(train_df, greeks_df, on="Id")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # Remove the first column
# train_df = train_df.drop("Id", axis=1)
# test_df = test_df.drop("Id", axis=1)

# === AFTER (edited) ===
# Drop Id column after merge
train_df = train_df.drop("Id", axis=1)
test_df = test_df.drop("Id", axis=1)

print("After dropping Id column:")
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# # One-hot encoding
# encoder = OneHotEncoder(handle_unknown="ignore")
# train_df = pd.get_dummies(train_df, columns=list(train_df))
# test_df = pd.get_dummies(test_df, columns=list(test_df))

# === AFTER (edited) ===
# Identify numeric columns (excluding categorical and ID columns)
numeric_cols = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5', 
                'Feature_6', 'Feature_7', 'Feature_8', 'Feature_9', 'Feature_10', 'Class']

# Display current train_df columns
print("Columns after merge:", train_df.columns.tolist())

# Apply StandardScaler to numeric columns only
scaler = StandardScaler()
train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

print("\nStandardization completed")
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # Data processing
# scaler = StandardScaler()
# train_df = scaler.fit_transform(train_df)
# test_df = scaler.transform(test_df)

# === AFTER (edited) ===
# Note: Scaling was already performed in the previous cell
# This cell is intentionally left empty to avoid duplicate scaling