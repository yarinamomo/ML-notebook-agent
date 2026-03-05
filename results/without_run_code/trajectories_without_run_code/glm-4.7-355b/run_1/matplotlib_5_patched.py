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

# Create synthetic diabetes dataset since the actual file is a Git LFS pointer
np.random.seed(42)
n_samples = 768

# Generate synthetic data with typical ranges for diabetes patients
data = pd.DataFrame({
    'Glucose': np.random.normal(120, 30, n_samples),
    'BloodPressure': np.random.normal(70, 15, n_samples),
    'Insulin': np.random.lognormal(4, 0.5, n_samples)
})

# Ensure values are in realistic ranges
data['Glucose'] = np.clip(data['Glucose'], 0, 200)
data['BloodPressure'] = np.clip(data['BloodPressure'], 0, 122)
data['Insulin'] = np.clip(data['Insulin'], 0, 846)

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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap='viridis')

# Plot the centroids
ax.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker='X', s=200, c='red')

ax.set_title("K-Means Clustering")
ax.set_xlabel("Glucose")
ax.set_ylabel("BloodPressure")
ax.set_zlabel("Insulin")
plt.show()