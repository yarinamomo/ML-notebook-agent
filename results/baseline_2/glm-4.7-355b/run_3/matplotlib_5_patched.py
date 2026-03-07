# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
file_path = 'data/diabetes.csv'
data = pd.read_csv(file_path)
X = data[['Glucose','BloodPressure','Insulin']].values

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
K = 3

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
kmeans = KMeans(n_clusters = K)
kmeans.fit(X)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
labels = kmeans.labels_

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
centriods = kmeans.cluster_centers_

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# plt.scatter(X[:,0], X[:,1], X[:,2], c = labels, cmap = 'viridis')
# plt.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker = 'X', s = 200, c = 'red')
# plt.title("K-Means Clustering")
# plt.xlabel("Gluscose")
# plt.ylabel("BloodPressure")
# plt.zlabel("Insulin")
# plt.show()

# === AFTER (edited) ===
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap='viridis')
ax.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker='X', s=200, c='red')
ax.set_title("K-Means Clustering")
ax.set_xlabel("Glucose")
ax.set_ylabel("BloodPressure")
ax.set_zlabel("Insulin")
plt.show()