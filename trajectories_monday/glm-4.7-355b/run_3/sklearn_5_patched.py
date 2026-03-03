# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df= pd.read_csv("data/iris.csv")

# === AFTER (edited) ===
from sklearn.datasets import load_iris

# Load the iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
X = df.drop("species",axis=1)
Y = df["species"]
x_train, x_test , y_train, y_test=train_test_split (X ,Y , test_size=0.25, random_state=42)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
clf = LogisticRegression()
clf.fit(x_train, y_train)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
y_pred = clf.predict(x_test)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# from sklearn import metrics
# confusion_matrix = metrics.confusion_matrix(y_test , y_pred)
# 
# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix , display_labels=[False,True])
# 
# cm_display.plot()
# plt.show()

# === AFTER (edited) ===
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(y_test , y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix , display_labels=['setosa', 'versicolor', 'virginica'])

cm_display.plot()
plt.show()