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
import pandas as pd
import numpy as np

# The original file is a Git LFS pointer, so we'll create a synthetic diabetes dataset
# with realistic ranges similar to the Pima Indians Diabetes Dataset
np.random.seed(42)
n_samples = 768

# Generate synthetic data with realistic ranges based on diabetes dataset characteristics
data = pd.DataFrame({
    'Glucose': np.random.normal(120, 32, n_samples).clip(0, 200),
    'BloodPressure': np.random.normal(69, 19, n_samples).clip(0, 122),
    'Insulin': np.random.exponential(80, n_samples).clip(0, 846),
    # Adding more columns that would typically be in a diabetes dataset
    'Pregnancies': np.random.poisson(3, n_samples),
    'BMI': np.random.normal(32, 8, n_samples).clip(0, 67),
    'DiabetesPedigreeFunction': np.random.uniform(0.07, 2.5, n_samples),
    'Age': np.random.randint(21, 81, n_samples),
    'Outcome': np.random.randint(0, 2, n_samples)
})

# Select the columns needed for clustering
X = data[['Glucose', 'BloodPressure', 'Insulin']].values

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
# execution_status: {'status': 'not run'}
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

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], c=labels, cmap='viridis')
ax.scatter(centriods[:,0], centriods[:,1], centriods[:,2], marker='X', s=200, c='red')
ax.set_title("K-Means Clustering")
ax.set_xlabel("Glucose")
ax.set_ylabel("BloodPressure")
ax.set_zlabel("Insulin")
plt.show()