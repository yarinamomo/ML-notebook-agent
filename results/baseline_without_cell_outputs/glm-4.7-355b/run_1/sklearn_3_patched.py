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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# # One-hot encoding
# encoder = OneHotEncoder(handle_unknown="ignore")
# train_df = pd.get_dummies(train_df, columns=list(train_df))
# test_df = pd.get_dummies(test_df, columns=list(test_df))

# === AFTER (edited) ===
# Identify categorical and numeric columns
categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train_df.select_dtypes(exclude=['object']).columns.tolist()

# Create preprocessing for numeric and categorical features
from sklearn.compose import ColumnTransformer

# StandardScaler for numeric columns
numeric_transformer = StandardScaler()

# OneHotEncoder for categorical columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply preprocessing to train and test data
X_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)

# The result is a numpy array, which is ready for model training
train_df = X_train
test_df = X_test

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)