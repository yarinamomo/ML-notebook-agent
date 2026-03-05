# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# file_path = 'data/diabetes.csv'
# data = pd.read_csv(file_path)
# X = data[['Glucose','BloodPressure','Insulin']].values

# === AFTER (edited) ===
file_path = 'data/diabetes.csv'
data = pd.read_csv(file_path)
print("Available columns:", data.columns.tolist())
# Try to find the correct columns (case-insensitive matching)
available_cols = data.columns.tolist()
X_cols = []
for col in ['Glucose', 'BloodPressure', 'Insulin']:
    # Find column with matching name (case-insensitive)
    match = [c for c in available_cols if c.lower() == col.lower()]
    if match:
        X_cols.append(match[0])
    else:
        print(f"Warning: {col} not found in columns")
if len(X_cols) < 3:
    print(f"Could not find all required columns. Using available columns: {available_cols[:3]}")
    X_cols = available_cols[:3]
X = data[X_cols].values

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
K = 3

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
kmeans = KMeans(n_clusters = K)
kmeans.fit(X)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
labels = kmeans.labels_

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
centriods = kmeans.cluster_centers_

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
plt.scatter(X[:,0], X[:,1], X[:,2], c = labels, cmap = 'viridis')
plt.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker = 'X', s = 200, c = 'red')
plt.title("K-Means Clustering")
plt.xlabel("Gluscose")
plt.ylabel("BloodPressure")
plt.zlabel("Insulin")
plt.show()