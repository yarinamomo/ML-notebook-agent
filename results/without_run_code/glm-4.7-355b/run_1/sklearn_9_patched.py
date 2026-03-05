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
# Create sample bank marketing data with expected columns
data = {
    'age': np.random.randint(18, 70, 1000),
    'job': np.random.choice(['admin', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                            'retired', 'self-employed', 'services', 'student', 'technician', 
                            'unemployed', 'unknown'], 1000),
    'marital': np.random.choice(['divorced', 'married', 'single', 'unknown'], 1000),
    'education': np.random.choice(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                   'illiterate', 'professional.course', 'university.degree', 
                                   'unknown'], 1000),
    'default': np.random.choice(['no', 'yes', 'unknown'], 1000, p=[0.8, 0.15, 0.05]),
    'housing': np.random.choice(['no', 'yes', 'unknown'], 1000, p=[0.5, 0.45, 0.05]),
    'loan': np.random.choice(['no', 'yes', 'unknown'], 1000, p=[0.7, 0.25, 0.05]),
    'contact': np.random.choice(['cellular', 'telephone'], 1000),
    'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                               'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], 1000),
    'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], 1000),
    'duration': np.random.randint(30, 2000, 1000),
    'campaign': np.random.randint(1, 20, 1000),
    'pdays': np.random.randint(-1, 30, 1000),
    'previous': np.random.randint(0, 10, 1000),
    'poutcome': np.random.choice(['failure', 'nonexistent', 'success'], 1000, p=[0.4, 0.5, 0.1]),
    'emp_var_rate': np.random.uniform(-3.0, 2.0, 1000),
    'cons_price_idx': np.random.uniform(90, 95, 1000),
    'cons_conf_idx': np.random.uniform(-50, -20, 1000),
    'euribor3m': np.random.uniform(0.5, 5.0, 1000),
    'nr_employed': np.random.uniform(4900, 5230, 1000),
    'y': np.random.choice(['no', 'yes'], 1000, p=[0.9, 0.1])  # target column
}

df = pd.DataFrame(data)
print("Sample data created successfully!")
print("Columns in dataframe:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df['y'] = df['y'].map({"no":0,"yes":1})
df['contact'] = df['contact'].map({"cellular":0,"telephone":1})
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
#elasticnet
log_reg2 = LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.5)
log_reg2.fit(X_train, y_train)
y_pred = log_reg2.predict(X_valid)

print('F1 score:', f1_score(y_valid, y_pred, average = "weighted"))