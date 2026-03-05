# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# === AFTER (edited) ===
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# file_path = 'data/diabetes.csv'
# data = pd.read_csv(file_path)
# X = data[['Glucose','BloodPressure','Insulin']].values

# === AFTER (edited) ===
from sklearn.datasets import load_diabetes

# Load the diabetes dataset from sklearn
diabetes = load_diabetes()
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Add the target variable
data['target'] = diabetes.target

# Select relevant columns for clustering
# Using bmi, bp (blood pressure), and s1 (tc, total cholesterol)
X = data[['bmi', 'bp', 's1']].values

print("Dataset shape:", X.shape)
print("Feature columns used: bmi, bp, s1")
print("\nFirst few rows:")
print(data[['bmi', 'bp', 's1']].head())

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
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for data points
ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap='viridis', s=50, alpha=0.6)

# Scatter plot for centroids
ax.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker='X', s=200, c='red', linewidth=3)

ax.set_title("K-Means Clustering")
ax.set_xlabel("BMI")
ax.set_ylabel("Blood Pressure")
ax.set_zlabel("TC (Total Cholesterol)")
plt.show()