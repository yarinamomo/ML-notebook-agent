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
# Search for train.csv in common locations
train_csv_path = None

# Check common directories
search_paths = [
    '/kaggle/input',
    '/kaggle/input/train.csv',
    'data/train.csv',
    './data',
    './'
]

# First try direct paths
for path in search_paths:
    if os.path.exists(path):
        if os.path.isfile(path):
            if path.endswith('train.csv'):
                train_csv_path = path
                break
        else:
            # Search in directory
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file == 'train.csv':
                        train_csv_path = os.path.join(root, file)
                        break
                if train_csv_path:
                    break
    if train_csv_path:
        break

# If not found, try current directory and subdirectories
if not train_csv_path:
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'train.csv':
                train_csv_path = os.path.join(root, file)
                break
        if train_csv_path:
            break

if train_csv_path:
    print(f"Found train.csv at: {train_csv_path}")
    data = pd.read_csv(train_csv_path)
else:
    raise ValueError("train.csv not found in any common location. Please ensure the dataset is available.")

data.head()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
data['Embarked'].fillna('S', inplace=True)
data.loc[data.Fare.isnull(),'Fare'] = data['Fare'].mean()
data.loc[data.Age.isnull(),'Age'] = data['Age'].mean()

labels = data["Survived"]
features = data.drop(columns=['PassengerId','Survived','PassengerId','Cabin','Ticket','Name'])

hot =pd.get_dummies(features,columns=['Sex','Embarked'])
features =hot
features,labels

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
X_train,X_test,y_train,y_test =train_test_split(features, labels, test_size=0.3)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
def evaluate(model,y_test=y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(predictions,y_test)
    return round(acc*100,3)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from pandas import Series

feature_importance = model.feature_importances_
Series_feat_imp = Series(feature_importance, index=data.columns)