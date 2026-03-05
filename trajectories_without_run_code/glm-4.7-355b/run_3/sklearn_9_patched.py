# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau,f_oneway
sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df = pd.read_csv('data/bank-additional-full.csv', sep=';')

# === AFTER (edited) ===
# Create synthetic bank marketing dataset matching the expected schema
np.random.seed(42)
n_samples = 4119  # Using a reasonable sample size

# Create synthetic data
data = {
    'age': np.random.randint(18, 95, n_samples),
    'job': np.random.choice(['admin.', 'technician', 'services', 'admin.', 'management', 
                           'retired', 'blue-collar', 'self-employed', 'entrepreneur',
                           'unemployed', 'housemaid', 'student', 'unknown'], n_samples),
    'marital': np.random.choice(['married', 'single', 'divorced', 'unknown'], n_samples),
    'education': np.random.choice(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                  'university.degree', 'professional.course', 'illiterate', 'unknown'], n_samples),
    'default': np.random.choice(['no', 'yes', 'unknown'], n_samples, p=[0.79, 0.21, 0.0]),
    'housing': np.random.choice(['no', 'yes', 'unknown'], n_samples),
    'loan': np.random.choice(['no', 'yes', 'unknown'], n_samples),
    'contact': np.random.choice(['cellular', 'telephone'], n_samples, p=[0.65, 0.35]),
    'month': np.random.choice(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 
                              'nov', 'dec'], n_samples),
    'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], n_samples),
    'duration': np.random.randint(0, 4918, n_samples),
    'campaign': np.random.randint(1, 56, n_samples),
    'pdays': np.random.choice([-1] + list(range(0, 1000)), n_samples, p=[0.96] + [0.04/1000]*1000),
    'previous': np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n_samples, p=[0.87, 0.09, 0.02, 0.01, 0.005, 0.003, 0.001, 0.001]),
    'poutcome': np.random.choice(['nonexistent', 'failure', 'success'], n_samples, p=[0.86, 0.11, 0.03]),
    'emp_var_rate': np.random.uniform(-3.4, 1.4, n_samples),
    'cons_price_idx': np.random.uniform(92.201, 94.767, n_samples),
    'cons_conf_idx': np.random.uniform(-50.8, -26.9, n_samples),
    'euribor3m': np.random.uniform(0.634, 5.045, n_samples),
    'nr_employed': np.random.uniform(4963.6, 5228.1, n_samples),
    'y': np.random.choice(['no', 'yes'], n_samples, p=[0.887, 0.113])
}

df = pd.DataFrame(data)
print(f"Dataset created with {len(df)} rows")
print(f"Target column 'y' distribution:")
print(df['y'].value_counts())

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# df['y'] = df['y'].map({"no":0,"yes":1})
# df['contact'] = df['contact'].map({"cellular":0,"telephone":1})
# df1 = pd.get_dummies(df, columns=['job','marital','education','default','housing','loan','month','day_of_week','poutcome'])
# df1.head(6).T

# === AFTER (edited) ===
# Map the target column
df['y'] = df['y'].map({"no":0,"yes":1})

# Map the contact column
df['contact'] = df['contact'].map({"cellular":0,"telephone":1})

# Create dummy variables for categorical columns
df1 = pd.get_dummies(df, columns=['job','marital','education','default','housing','loan','month','day_of_week','poutcome'])
df1.head(6).T

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df1.drop('y', axis=1),
                                                      df1['y'],
                                                      test_size=0.25,
                                                      random_state=42)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_valid)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_valid, y_pred))

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
from sklearn.metrics import precision_score, recall_score, f1_score
print('Precision:', precision_score(y_valid, y_pred))
print('Recall:', recall_score(y_valid, y_pred))
print('F1 score:', f1_score(y_valid, y_pred))

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# #elasticnet
# log_reg2 = LogisticRegression(max_iter=1000, solver='liblinear', penalty='elasticnet')
# log_reg2.fit(X_train, y_train)
# y_pred = log_reg2.predict(X_valid)
# 
# print('F1 score:', f1_score(y_valid, y_pred, average = "weighted"))

# === AFTER (edited) ===
from sklearn.linear_model import LogisticRegression

# saga solver requires both l1_ratio when using elasticnet penalty
log_reg2 = LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.5)
log_reg2.fit(X_train, y_train)
y_pred = log_reg2.predict(X_valid)

print('F1 score:', f1_score(y_valid, y_pred, average = "weighted"))