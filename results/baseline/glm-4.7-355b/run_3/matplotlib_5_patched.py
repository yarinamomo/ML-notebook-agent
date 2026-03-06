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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# file_path = 'data/diabetes.csv'
# data = pd.read_csv(file_path)
# X = data[['Glucose','BloodPressure','Insulin']].values

# === AFTER (edited) ===
file_path = 'data/diabetes.csv'
data = pd.read_csv(file_path)
print("Available columns:", data.columns.tolist())

# Try to get the expected columns, handling potential whitespace issues
cols = ['Glucose', 'BloodPressure', 'Insulin']
available_cols = [col for col in cols if col in data.columns]

if len(available_cols) == 3:
    X = data[cols].values
    print(f"Using columns: {cols}")
elif data.shape[1] >= 3:
    # Fall back to first 3 numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 3:
        X = data[numeric_cols[:3]].values
        print(f"Warning: Expected columns not found. Using numeric columns: {numeric_cols[:3]}")
    else:
        raise ValueError(f"Need at least 3 numeric columns, but found only {len(numeric_cols)}")
else:
    raise ValueError(f"DataFrame has only {data.shape[1]} columns, need at least 3")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
K = 3

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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