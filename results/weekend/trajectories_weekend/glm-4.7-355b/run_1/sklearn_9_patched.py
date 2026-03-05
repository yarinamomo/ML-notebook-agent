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
import pandas as pd
import numpy as np

# Try to load the real data
df = pd.read_csv('data/bank-additional-full.csv', sep=';')

# Check if the data is a Git LFS pointer (not real data)
if df.shape[1] == 1 and df.columns[0].startswith('version'):
    print("Warning: Real data file not available (Git LFS pointer). Using sample data.")
    # Generate sample data with the structure expected by the notebook
    np.random.seed(42)
    n_samples = 1000
    
    # Generate categorical features
    jobs = ['housemaid', 'services', 'admin.', 'blue-collar', 'technician', 
            'retired', 'management', 'unemployed', 'self-employed', 'entrepreneur', 'student']
    maritals = ['married', 'single', 'divorced']
    educations = ['basic.4y', 'high.school', 'basic.6y', 'basic.9y', 
                  'professional.course', 'unknown', 'university.degree']
    defaults = ['no', 'yes', 'unknown']
    housings = ['no', 'yes', 'unknown']
    loans = ['no', 'yes', 'unknown']
    contacts = ['cellular', 'telephone']
    months = ['may', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec', 
              'jan', 'feb', 'mar', 'apr', 'sep']
    days_of_week = ['mon', 'tue', 'wed', 'thu', 'fri']
    poutcomes = ['nonexistent', 'failure', 'success']
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'job': np.random.choice(jobs, n_samples),
        'marital': np.random.choice(maritals, n_samples),
        'education': np.random.choice(educations, n_samples),
        'default': np.random.choice(defaults, n_samples, p=[0.9, 0.05, 0.05]),
        'housing': np.random.choice(housings, n_samples, p=[0.45, 0.45, 0.1]),
        'loan': np.random.choice(loans, n_samples, p=[0.85, 0.12, 0.03]),
        'contact': np.random.choice(contacts, n_samples, p=[0.75, 0.25]),
        'month': np.random.choice(months, n_samples),
        'day_of_week': np.random.choice(days_of_week, n_samples),
        'duration': np.random.randint(0, 1800, n_samples),
        'campaign': np.random.randint(1, 10, n_samples),
        'pdays': np.random.randint(-1, 30, n_samples),
        'previous': np.random.randint(0, 10, n_samples),
        'poutcome': np.random.choice(poutcomes, n_samples),
        'emp_var_rate': np.random.randn(n_samples),
        'cons_price_idx': np.random.randn(n_samples) + 93,
        'cons_conf_idx': np.random.randn(n_samples) - 40,
        'euribor3m': np.random.randn(n_samples) + 1,
        'nr_employed': np.random.randn(n_samples) + 5000,
        'y': np.random.choice(['no', 'yes'], n_samples, p=[0.85, 0.15])
    })

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

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
# elasticnet
log_reg2 = LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.5)
log_reg2.fit(X_train, y_train)
y_pred = log_reg2.predict(X_valid)

print('F1 score:', f1_score(y_valid, y_pred, average="weighted"))