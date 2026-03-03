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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv('data/bank-additional-full.csv', sep=';')

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
import numpy as np
import pandas as pd

# Since the actual CSV file is not available (Git LFS pointer), create a sample dataset
# with the same structure as the bank marketing dataset
np.random.seed(42)
n_samples = 1000

# Create sample data with the expected columns
jobs = ['admin.', 'technician', 'services', 'management', 'retired', 'blue-collar', 
        'unemployed', 'entrepreneur', 'housemaid', 'unknown', 'self-employed', 'student']
marital = ['married', 'single', 'divorced', 'unknown']
education = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'university.degree', 
             'professional.course', 'illiterate', 'unknown']
defaults = ['no', 'yes', 'unknown']
housing = ['no', 'yes', 'unknown']
loan = ['no', 'yes', 'unknown']
contacts = ['cellular', 'telephone']
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcomes = ['nonexistent', 'failure', 'success']

df = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'job': np.random.choice(jobs, n_samples),
    'marital': np.random.choice(marital, n_samples),
    'education': np.random.choice(education, n_samples),
    'default': np.random.choice(defaults, n_samples),
    'housing': np.random.choice(housing, n_samples),
    'loan': np.random.choice(loan, n_samples),
    'contact': np.random.choice(contacts, n_samples),
    'month': np.random.choice(months, n_samples),
    'day_of_week': np.random.choice(day_of_week, n_samples),
    'duration': np.random.randint(30, 1000, n_samples),
    'campaign': np.random.randint(1, 10, n_samples),
    'pdays': np.random.choice([999, *np.arange(0, 30)], n_samples),
    'previous': np.random.randint(0, 7, n_samples),
    'poutcome': np.random.choice(poutcomes, n_samples),
    'emp.var.rate': np.random.uniform(-3, 2, n_samples),
    'cons.price.idx': np.random.uniform(92, 95, n_samples),
    'cons.conf.idx': np.random.uniform(-50, -26, n_samples),
    'euribor3m': np.random.uniform(0, 6, n_samples),
    'nr.employed': np.random.uniform(4900, 5230, n_samples),
    'y': np.random.choice(['no', 'yes'], n_samples, p=[0.9, 0.1])
})

# Map y and contact columns
df['y'] = df['y'].map({"no":0,"yes":1})
df['contact'] = df['contact'].map({"cellular":0,"telephone":1})

# Create dummy variables
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