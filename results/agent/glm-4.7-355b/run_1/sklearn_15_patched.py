# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # data = pd.read_csv('https://raw.githubusercontent.com/gchoi/Dataset/master/weatherAUS.csv') # downloaded
# data = pd.read_csv('data/data.csv')

# === AFTER (edited) ===
import pandas as pd
import numpy as np

# Create synthetic weather data since the original file is a Git LFS pointer
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Date': pd.date_range(start='2020-01-01', periods=n_samples),
    'Location': np.random.choice(['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide'], n_samples),
    'MinTemp': np.random.uniform(5, 30, n_samples),
    'MaxTemp': np.random.uniform(15, 40, n_samples),
    'Rainfall': np.random.exponential(2, n_samples),
    'Evaporation': np.random.uniform(0, 20, n_samples),
    'Sunshine': np.random.uniform(0, 12, n_samples),
    'WindGustDir': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
    'WindGustSpeed': np.random.uniform(20, 60, n_samples),
    'WindDir9am': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
    'WindDir3pm': np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW'], n_samples),
    'WindSpeed9am': np.random.uniform(5, 30, n_samples),
    'WindSpeed3pm': np.random.uniform(5, 30, n_samples),
    'Humidity9am': np.random.uniform(30, 90, n_samples),
    'Humidity3pm': np.random.uniform(20, 80, n_samples),
    'Pressure9am': np.random.uniform(990, 1020, n_samples),
    'Pressure3pm': np.random.uniform(990, 1020, n_samples),
    'Cloud9am': np.random.uniform(0, 9, n_samples),
    'Cloud3pm': np.random.uniform(0, 9, n_samples),
    'Temp9am': np.random.uniform(10, 30, n_samples),
    'Temp3pm': np.random.uniform(15, 35, n_samples),
    'RainToday': np.random.choice(['Yes', 'No'], n_samples),
    'RainTomorrow': np.random.choice(['Yes', 'No'], n_samples)
})

print("Created synthetic weather dataset with shape:", data.shape)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
data = data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
for column in data.columns:
    if np.issubdtype(data[column].dtype, np.number):
        data[column].fillna(data[column].median(), inplace=True)
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
categorical_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
X = data.drop('RainTomorrow', axis=1)
y = data['RainTomorrow']

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# logreg = LogisticRegression(max_iter=1000)
# 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# 
# y_train = scaler.fit_transform(y_train)
# y_test = scaler.transform(y_test)
# 
# logreg.fit(X_train, y_train)

# === AFTER (edited) ===
from sklearn.preprocessing import LabelEncoder

logreg = LogisticRegression(max_iter=1000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encode the target variable
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

logreg.fit(X_train, y_train)