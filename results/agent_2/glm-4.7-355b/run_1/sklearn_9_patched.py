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
from sklearn.metrics import precision_score, recall_score, f1_score
log_reg2 = LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.5)
log_reg2.fit(X_train, y_train)
y_pred = log_reg2.predict(X_valid)

print('F1 score:', f1_score(y_valid, y_pred, average = "weighted"))