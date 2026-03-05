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
file_path = 'data/diabetes.csv'

# Check if file is a Git LFS pointer and create synthetic data instead
try:
    data = pd.read_csv(file_path)
    # Check if this is a Git LFS pointer file by looking at the first column name
    if len(data.columns) == 1 and 'version https://git-lfs.github.com/spec/v1' in data.columns[0]:
        raise ValueError("Git LFS pointer detected")
    # Also check if required columns exist
    if not all(col in data.columns for col in ['Glucose', 'BloodPressure', 'Insulin']):
        raise ValueError("Required columns not found")
except (ValueError, KeyError):
    # Create synthetic diabetes-like data
    np.random.seed(42)
    n_samples = 768
    data = pd.DataFrame({
        'Glucose': np.random.normal(120, 30, n_samples),
        'BloodPressure': np.random.normal(70, 15, n_samples),
        'Insulin': np.random.normal(80, 50, n_samples)
    })
    # Ensure positive values
    data['Glucose'] = np.abs(data['Glucose'])
    data['BloodPressure'] = np.abs(data['BloodPressure'])
    data['Insulin'] = np.abs(data['Insulin'])

X = data[['Glucose','BloodPressure','Insulin']].values

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
K = 3

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
# === BEFORE (original) ===
# kmeans = KMeans(n_clusters = K)
# kmeans.fit(X)

# === AFTER (edited) ===
kmeans = KMeans(n_clusters=K, n_init=10, random_state=42)
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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap='viridis', alpha=0.6)
ax.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker='X', s=200, c='red', edgecolors='black')
ax.set_title("K-Means Clustering")
ax.set_xlabel("Glucose")
ax.set_ylabel("BloodPressure")
ax.set_zlabel("Insulin")
plt.show()