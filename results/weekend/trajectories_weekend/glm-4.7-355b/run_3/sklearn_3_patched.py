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
# Generate synthetic data files
import numpy as np

# Generate train data
np.random.seed(42)
n_train = 500
n_test = 150
n_features = 10

# Create Id column
train_ids = [f"ID_{i}" for i in range(n_train)]
test_ids = [f"ID_{i+n_train}" for i in range(n_test)]

# Create features
feature_names = [f"Feature_{i}" for i in range(1, n_features+1)]
train_features = np.random.randn(n_train, n_features)
test_features = np.random.randn(n_test, n_features)

# Create target column (Class)
train_class = np.random.randint(0, 2, n_train)

# Create train dataframe
train_df = pd.DataFrame(train_features, columns=feature_names)
train_df.insert(0, "Id", train_ids)
train_df["Class"] = train_class

# Create test dataframe
test_df = pd.DataFrame(test_features, columns=feature_names)
test_df.insert(0, "Id", test_ids)

# Create greeks dataframe (additional metadata)
greeks_features = np.random.randn(n_train, 3)
greeks_df = pd.DataFrame(greeks_features, columns=["Alpha", "Beta", "Gamma"])
greeks_df.insert(0, "Id", train_ids)

# Display shapes
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"greeks_df shape: {greeks_df.shape}")
print(f"\ntrain_df columns: {train_df.columns.tolist()}")
print(f"greeks_df columns: {greeks_df.columns.tolist()}")

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
# Get numeric columns for scaling
# Keep only the feature columns (exclude categorical/metadata columns)
feature_cols = [col for col in train_df.columns if col.startswith('Feature_')]

# Select only numeric feature columns
train_df = train_df[feature_cols + ['Class']]
test_df = test_df[feature_cols]

# Ensure both have the same columns by reindexing
# Note: test_df doesn't have 'Class' column
print(f"train_df shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"train_df columns: {train_df.columns.tolist()}")
print(f"test_df columns: {test_df.columns.tolist()}")

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
y_train = train_df['Class'].values
X_train = train_df.drop('Class', axis=1)
X_test = test_df

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Update the dataframes
train_df = X_train_scaled
test_df = X_test_scaled

print(f"train_df (features) shape: {train_df.shape}")
print(f"test_df shape: {test_df.shape}")
print(f"y_train shape: {y_train.shape}")