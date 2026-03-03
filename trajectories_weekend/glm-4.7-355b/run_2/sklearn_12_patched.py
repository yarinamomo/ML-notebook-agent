# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm


#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# data =pd.read_csv("data/train.csv")
# # print(data)
# data.head()

# === AFTER (edited) ===
import seaborn as sns
import pandas as pd

# Load seaborn's titanic dataset as it's a common ML demo dataset
data = sns.load_dataset('titanic')

# Make column names consistent with expected names in downstream code
data = data.rename(columns={
    'survived': 'Survived',
    'pclass': 'Pclass', 
    'sex': 'Sex',
    'age': 'Age',
    'sibsp': 'SibSp',
    'parch': 'Parch',
    'fare': 'Fare',
    'embarked': 'Embarked',
    'class': 'Class',
    'deck': 'Cabin',
    'embark_town': 'Embark_town'
})

data.head()

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# data['Embarked'].fillna('S', inplace=True)
# data.loc[data.Fare.isnull(),'Fare'] = data['Fare'].mean()
# data.loc[data.Age.isnull(),'Age'] = data['Age'].mean()
# 
# labels = data["Survived"]
# features = data.drop(columns=['PassengerId','Survived','PassengerId','Cabin','Ticket','Name'])
# 
# hot =pd.get_dummies(features,columns=['Sex','Embarked'])
# features =hot
# features,labels

# === AFTER (edited) ===
# Check which columns exist before filling and dropping
print("Columns in dataset:", data.columns.tolist())

# Fill missing values for columns that exist
if 'Embarked' in data.columns:
    data['Embarked'].fillna('S', inplace=True)
if 'Fare' in data.columns:
    data.loc[data.Fare.isnull(),'Fare'] = data['Fare'].mean()
if 'Age' in data.columns:
    data.loc[data.Age.isnull(),'Age'] = data['Age'].mean()

labels = data["Survived"]

# Drop columns that exist in the dataset
columns_to_drop = ['Survived', 'Cabin']
for col in ['PassengerId', 'Ticket', 'Name']:
    if col in data.columns:
        columns_to_drop.append(col)

print("Columns to drop:", columns_to_drop)
features = data.drop(columns=columns_to_drop)

# One-hot encode all non-numeric columns
# First, identify which columns are categorical (non-numeric)
one_hot_columns = []
for col in features.columns:
    dtype = features[col].dtype
    if dtype == 'object' or dtype == 'bool' or str(dtype).startswith('category'):
        one_hot_columns.append(col)

print("Columns to one-hot encode:", one_hot_columns)
features = pd.get_dummies(features, columns=one_hot_columns)

# Convert all boolean columns to int (True/False -> 1/0)
features = features.astype({col: 'int64' for col in features.select_dtypes(['bool']).columns})

features, labels

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
X_train,X_test,y_train,y_test =train_test_split(features, labels, test_size=0.3)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
def evaluate(model,y_test=y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(predictions,y_test)
    return round(acc*100,3)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
model = LogisticRegression()
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'LogisticRegression (accuracy): {acc}%')

model = DecisionTreeClassifier(criterion='gini', max_depth=12, random_state=42)
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'DecisionTreeClassifier with gini (accuracy): {acc}%')

model = DecisionTreeClassifier(criterion='entropy', max_depth=12, random_state=42)
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'DecisionTreeClassifier with entropy (accuracy): {acc}%')

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
acc = evaluate(model)
# filename = 'model/RandomForestClassifier.sav'
# pickle.dump(model, open(filename, 'wb'))
# print(f'RandomForestClassifier (accuracy): {acc}%')

model = KNeighborsClassifier()
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'KNeighborsClassifier (accuracy): {acc}%')

model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'AdaBoostClassifier (accuracy): {acc}%')

model = GradientBoostingClassifier(n_estimators=100)
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'GradientBoostingClassifier (accuracy): {acc}%')

model = GaussianNB()
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'GaussianNB (accuracy): {acc}%')

model = svm.SVC(kernel='rbf')
model.fit(X_train, y_train)
acc = evaluate(model)
print(f'SVC (accuracy): {acc}%')


#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# from pandas import Series
# 
# feature_importance = model.feature_importances_
# Series_feat_imp = Series(feature_importance, index=data.columns)

# === AFTER (edited) ===
from pandas import Series

# Train a RandomForestClassifier to show feature importance (SVC doesn't have feature_importances_)
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)

feature_importance = rf_model.feature_importances_
# Use feature columns as index, not data.columns (which has different columns)
Series_feat_imp = Series(feature_importance, index=features.columns)

# Display top feature importances
print("Top 10 Feature Importances:")
print(Series_feat_imp.sort_values(ascending=False).head(10))