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
# Create synthetic data since the CSV files are Git LFS pointer files
np.random.seed(42)

# Create train data with Id column (no Alpha/Beta yet)
n_train = 100
train_df = pd.DataFrame({
    'Id': [f'id_{i}' for i in range(n_train)],
    'Age': np.random.randint(18, 80, n_train),
    'BMI': np.random.uniform(18, 35, n_train),
    'BloodPressure': np.random.randint(90, 140, n_train),
    'Cholesterol': np.random.randint(150, 300, n_train),
    'Class': np.random.randint(0, 2, n_train)
})

# Create test data (also no Alpha/Beta - they should come from greeks-like data)
n_test = 50
test_df = pd.DataFrame({
    'Id': [f'id_{i+n_train}' for i in range(n_test)],
    'Age': np.random.randint(18, 80, n_test),
    'BMI': np.random.uniform(18, 35, n_test),
    'BloodPressure': np.random.randint(90, 140, n_test),
    'Cholesterol': np.random.randint(150, 300, n_test),
    'Class': np.random.randint(0, 2, n_test)
})

# Create greeks data with Id column and categorical features
greeks_df = pd.DataFrame({
    'Id': [f'id_{i}' for i in range(n_train)],
    'Alpha': np.random.choice(['A', 'B', 'C'], n_train),
    'Beta': np.random.choice(['+', '-', '0'], n_train)
})

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
# One-hot encode only the categorical columns
categorical_cols = ['Alpha', 'Beta']

# Check which categorical columns exist in each dataframe
train_cat_cols = [col for col in categorical_cols if col in train_df.columns]
test_cat_cols = [col for col in categorical_cols if col in test_df.columns]

# Perform one-hot encoding only on categorical columns
if train_cat_cols:
    train_df_encoded = pd.get_dummies(train_df, columns=train_cat_cols)
else:
    train_df_encoded = train_df.copy()

if test_cat_cols:
    test_df_encoded = pd.get_dummies(test_df, columns=test_cat_cols)
else:
    test_df_encoded = test_df.copy()

# Align columns between train and test
# Get all columns from both
all_columns = set(train_df_encoded.columns) | set(test_df_encoded.columns)

# Add missing columns to each DataFrame
for col in all_columns:
    if col not in train_df_encoded.columns:
        train_df_encoded[col] = 0
    if col not in test_df_encoded.columns:
        test_df_encoded[col] = 0

# Ensure same column order
train_df = train_df_encoded[sorted(train_df_encoded.columns)]
test_df = test_df_encoded[sorted(test_df_encoded.columns)]

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)