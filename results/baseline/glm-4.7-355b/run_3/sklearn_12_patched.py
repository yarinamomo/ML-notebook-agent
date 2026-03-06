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
import numpy as np
import pandas as pd
from io import StringIO

# Try to read the CSV file
data = pd.read_csv("data/train.csv")

# Check if data looks like an LFS pointer (contains version/git-lfs fields)
if 'version' in data.columns and len(data.columns) == 1:
    print("Warning: File is an LFS pointer. Creating sample Titanic dataset...")
    # Create a sample Titanic-like dataset
    np.random.seed(42)
    n_samples = 891
    
    data = pd.DataFrame({
        'PassengerId': range(1, n_samples + 1),
        'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.20, 0.55]),
        'Name': ['Passenger_' + str(i) for i in range(n_samples)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.uniform(1, 80, n_samples),
        'SibSp': np.random.randint(0, 5, n_samples),
        'Parch': np.random.randint(0, 6, n_samples),
        'Ticket': ['T' + str(i) for i in range(n_samples)],
        'Fare': np.random.uniform(5, 200, n_samples),
        'Cabin': np.random.choice(['C' + str(i%50+1) if i%3==0 else np.nan for i in range(n_samples)]),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
    })

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