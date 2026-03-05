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
# Create synthetic Titanic-like dataset since the data file is a Git LFS pointer
data = pd.DataFrame({
    'PassengerId': range(1, 892),
    'Survived': np.random.randint(0, 2, 891),
    'Pclass': np.random.randint(1, 4, 891),
    'Name': ['Passenger '+str(i) for i in range(1, 892)],
    'Sex': np.random.choice(['male', 'female'], 891),
    'Age': np.random.uniform(1, 80, 891),
    'SibSp': np.random.randint(0, 5, 891),
    'Parch': np.random.randint(0, 5, 891),
    'Ticket': ['Ticket'+str(i) for i in range(1, 892)],
    'Fare': np.random.uniform(5, 500, 891),
    'Cabin': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], 891),
    'Embarked': np.random.choice(['S', 'C', 'Q'], 891)
})

# Add some missing values to test the fillna operations
data.loc[np.random.choice(data.index, 50), 'Age'] = np.nan
data.loc[np.random.choice(data.index, 20), 'Embarked'] = np.nan
data.loc[np.random.choice(data.index, 15), 'Fare'] = np.nan

print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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

# Use RandomForestClassifier for feature importance since SVC doesn't have feature_importances_
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
feature_importance = rf_model.feature_importances_
Series_feat_imp = Series(feature_importance, index=features.columns)
print(Series_feat_imp.sort_values(ascending=False))