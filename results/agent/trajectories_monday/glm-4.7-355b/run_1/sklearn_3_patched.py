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
# Create synthetic data since actual CSV files are not available
np.random.seed(42)

# Synthetic training data with Id column
train_df = pd.DataFrame({
    'Id': [f'train_{i}' for i in range(100)],
    'Feature_1': np.random.randn(100),
    'Feature_2': np.random.randn(100),
    'Feature_3': np.random.randn(100),
    'Class': np.random.randint(0, 2, 100)
})

# Synthetic test data with Id column
test_df = pd.DataFrame({
    'Id': [f'test_{i}' for i in range(50)],
    'Feature_1': np.random.randn(50),
    'Feature_2': np.random.randn(50),
    'Feature_3': np.random.randn(50)
})

# Synthetic Greeks data with Id column
greeks_df = pd.DataFrame({
    'Id': [f'train_{i}' for i in range(100)],
    'Alpha': np.random.choice(['A', 'B', 'C', 'D'], 100),
    'Beta': np.random.choice(['A', 'B', 'C'], 100),
    'Gamma': np.random.choice(['A', 'B', 'C', 'D', 'E'], 100),
    'Delta': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 100)
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
# One-hot encode categorical columns only
categorical_cols = ['Alpha', 'Beta', 'Gamma', 'Delta']
numeric_cols = ['Feature_1', 'Feature_2', 'Feature_3', 'Class']

# Only encode categorical columns that exist
train_df = pd.get_dummies(train_df, columns=[c for c in categorical_cols if c in train_df.columns], drop_first=False)
test_df = pd.get_dummies(test_df, columns=[c for c in categorical_cols if c in test_df.columns], drop_first=False)

# Ensure both train and test have the same columns after encoding
all_columns = set(train_df.columns) | set(test_df.columns)
for col in all_columns:
    if col not in train_df.columns:
        train_df[col] = 0
    if col not in test_df.columns:
        test_df[col] = 0

# Reorder columns to match
test_df = test_df[train_df.columns.drop('Class')] if 'Class' in train_df.columns else test_df

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
# Separate features and target
y_train = train_df['Class']
X_train = train_df.drop('Class', axis=1)
X_test = test_df

# Scale only the features (not the label)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Update train_df and test_df
train_df = X_train_scaled
test_df = X_test_scaled

print("Data scaled successfully")
print(f"Train features shape: {train_df.shape}")
print(f"Test features shape: {test_df.shape}")