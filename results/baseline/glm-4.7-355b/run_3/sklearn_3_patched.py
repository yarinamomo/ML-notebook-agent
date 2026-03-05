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
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# train_df = pd.merge(train_df, greeks_df, on="Id")

# === AFTER (edited) ===
# Check what columns exist in greeks_df
print("Greeks df columns:", greeks_df.columns.tolist())
print("\nTrain df columns:", train_df.columns.tolist())

# Try different column name options
if 'Id' in train_df.columns and 'Id' in greeks_df.columns:
    train_df = pd.merge(train_df, greeks_df, on="Id")
elif 'Id' in train_df.columns and 'id' in greeks_df.columns:
    train_df = pd.merge(train_df, greeks_df, left_on="Id", right_on="id")
elif 'id' in train_df.columns and 'id' in greeks_df.columns:
    train_df = pd.merge(train_df, greeks_df, on="id")
elif 'Id' in train_df.columns:
    # If greeks_df doesn't have Id, merge by index
    train_df = train_df.reset_index(drop=True)
    greeks_df = greeks_df.reset_index(drop=True)
    train_df = pd.concat([train_df, greeks_df], axis=1)
else:
    # If train_df doesn't have Id column, try lowercase
    train_df = pd.merge(train_df, greeks_df, on="id")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# Remove the first column
train_df = train_df.drop("Id", axis=1)
test_df = test_df.drop("Id", axis=1)


#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# One-hot encoding
encoder = OneHotEncoder(handle_unknown="ignore")
train_df = pd.get_dummies(train_df, columns=list(train_df))
test_df = pd.get_dummies(test_df, columns=list(test_df))

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# Data processing
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)