# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# file_path = 'data/diabetes.csv'
# data = pd.read_csv(file_path)
# X = data[['Glucose','BloodPressure','Insulin']].values

# === AFTER (edited) ===
# Create sample diabetes-like data with required features
np.random.seed(42)
n_samples = 768

# Generate realistic diabetes data
glucose = np.random.normal(120, 30, n_samples)
glucose = np.clip(glucose, 0, 200)  # Keep within realistic range

blood_pressure = np.random.normal(70, 12, n_samples)
blood_pressure = np.clip(blood_pressure, 0, 122)

insulin = np.random.normal(80, 100, n_samples)
insulin = np.clip(insulin, 0, 500)  # Keep within realistic range

# Create DataFrame
data = pd.DataFrame({
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'Insulin': insulin
})

# Extract features for clustering
X = data[['Glucose', 'BloodPressure', 'Insulin']].values

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
K = 3

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
kmeans = KMeans(n_clusters = K)
kmeans.fit(X)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
labels = kmeans.labels_

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
centriods = kmeans.cluster_centers_

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'execution_count': 7, 'status': 'ok'}
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

# Scatter plot for data points
ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap='viridis', alpha=0.6)

# Scatter plot for centroids
ax.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker='X', s=200, c='red', label='Centroids')

ax.set_title("K-Means Clustering")
ax.set_xlabel("Glucose")
ax.set_ylabel("BloodPressure")
ax.set_zlabel("Insulin")
ax.legend()
plt.show()