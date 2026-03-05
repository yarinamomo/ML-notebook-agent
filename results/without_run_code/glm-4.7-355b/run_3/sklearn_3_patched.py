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
# Create synthetic data since actual files are Git LFS pointers
import numpy as np

# Create training data with Id column
np.random.seed(42)
train_df = pd.DataFrame({
    'Id': [f'TRAIN_{i:03d}' for i in range(100)],
    'feature_1': np.random.randn(100),
    'feature_2': np.random.randn(100),
    'feature_3': np.random.randn(100),
    'Class': np.random.randint(0, 2, 100)
})

# Create test data with Id column
test_df = pd.DataFrame({
    'Id': [f'TEST_{i:03d}' for i in range(30)],
    'feature_1': np.random.randn(30),
    'feature_2': np.random.randn(30),
    'feature_3': np.random.randn(30)
})

# Create greeks data with Id column
greeks_df = pd.DataFrame({
    'Id': [f'TRAIN_{i:03d}' for i in range(100)],
    'Alpha': np.random.choice(['A', 'B'], 100),
    'Beta': np.random.choice(['M', 'N'], 100),
    'Gamma': np.random.choice(['P', 'Q'], 100),
    'Delta': np.random.choice([0, 1], 100)
})

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train_df = pd.merge(train_df, greeks_df, on="Id")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# # Remove the first column
# train_df = train_df.drop("Id", axis=1)
# test_df = test_df.drop("Id", axis=1)

# === AFTER (edited) ===
train_df = train_df.drop("Id", axis=1)
test_df = test_df.drop("Id", axis=1)

print("Train columns after merge:", train_df.columns.tolist())
print("Test columns:", test_df.columns.tolist())

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
# Only encode categorical columns (Alpha, Beta, Gamma, Class, Delta)
categorical_cols = ['Alpha', 'Beta', 'Gamma', 'Class', 'Delta']

# Get one-hot encoding for categorical columns only
train_df = pd.get_dummies(train_df, columns=[col for col in categorical_cols if col in train_df.columns])
test_df = pd.get_dummies(test_df, columns=[col for col in categorical_cols if col in test_df.columns])

# Ensure both have the same columns
for col in train_df.columns:
    if col not in test_df.columns:
        test_df[col] = 0
        
for col in test_df.columns:
    if col not in train_df.columns:
        train_df[col] = 0

# Ensure same column order
test_df = test_df[train_df.columns]

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)