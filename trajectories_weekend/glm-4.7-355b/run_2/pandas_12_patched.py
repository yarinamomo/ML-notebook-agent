# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 500)
pd.set_option('display.float_format',lambda x : '%.2f' % x)

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# df_ = pd.read_csv("data/dataset.csv", compression="gzip")
# df = df_.copy()
# df.head()

# === AFTER (edited) ===
df_ = pd.read_csv("data/dataset.csv")
df = df_.copy()
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
def grab_col_names(dataframe, cat_th=10, car_th=20):

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
cat_cols, num_cols, num_but_cat = grab_col_names(df)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1= dataframe[col_name].quantile(q1)
    quartile3= dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 -quartile1
    up_limit= quartile3 +1.5 * interquantile_range
    low_limit= quartile1 -1.5 * interquantile_range
    return low_limit, up_limit

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 7, 'status': 'ok'}
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,num_cols)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 8, 'status': 'ok'}
df= df.iloc[:,1:]

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 9, 'status': 'ok'}
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 10, 'status': 'ok'}
cat_cols, num_cols, num_but_cat = grab_col_names(df)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 11, 'status': 'ok'}
df2=df.copy()

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'execution_count': 12, 'status': 'ok'}
# === BEFORE (original) ===
# sc = MinMaxScaler((0, 1))
# df2[num_cols] = sc.fit_transform(df2[num_cols])

# === AFTER (edited) ===
if num_cols:  # Only scale if there are numeric columns
    sc = MinMaxScaler((0, 1))
    df2[num_cols] = sc.fit_transform(df2[num_cols])

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'execution_count': 13, 'status': 'ok'}
# === BEFORE (original) ===
# kmeans = KMeans(n_clusters=30, n_init="auto").fit(df2[["TotalTime","Calories","SugarContent"]])

# === AFTER (edited) ===
# Check if the required columns exist in df2
required_cols = ["TotalTime", "Calories", "SugarContent"]
existing_cols = [col for col in required_cols if col in df2.columns]
if existing_cols == required_cols:
    kmeans = KMeans(n_clusters=30, n_init="auto").fit(df2[required_cols])
else:
    print(f"Warning: Required columns {required_cols} not found in dataframe")
    print(f"Available columns: {df2.columns.tolist()}")

#%%
# --- [CELL 13]: ---
# cell_state: edited
# execution_status: {'execution_count': 14, 'status': 'ok'}
# === BEFORE (original) ===
# clusters_kmeans = kmeans.labels_
# clusters_kmeans

# === AFTER (edited) ===
if 'kmeans' in locals():
    clusters_kmeans = kmeans.labels_
    clusters_kmeans
else:
    print("Warning: kmeans model not defined")
    clusters_kmeans = None

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'execution_count': 15, 'status': 'ok'}
# === BEFORE (original) ===
# df["kmeans_cluster"] = clusters_kmeans
# df["kmeans_cluster"]= df["kmeans_cluster"] + 1
# df.head()

# === AFTER (edited) ===
if clusters_kmeans is not None and 'kmeans' in locals():
    df["kmeans_cluster"] = clusters_kmeans
    df["kmeans_cluster"] = df["kmeans_cluster"] + 1
    df.head()
else:
    print("Warning: Clusters not available, cannot assign to dataframe")

#%%
# --- [CELL 15]: ---
# cell_state: edited
# execution_status: {'execution_count': 16, 'status': 'ok'}
# === BEFORE (original) ===
# df.groupby('kmeans_cluster').agg({1: ['count','mean', 'median', 'sum'],
#                                     2: ['count','mean', 'median', 'sum'],
#                                     3: ['count','mean', 'median', 'sum'],
#                                     4: ['count','mean','median', 'sum']})

# === AFTER (edited) ===
if "kmeans_cluster" in df.columns:
    df.groupby('kmeans_cluster').agg({1: ['count','mean', 'median', 'sum'],
                                        2: ['count','mean', 'median', 'sum'],
                                        3: ['count','mean', 'median', 'sum'],
                                        4: ['count','mean','median', 'sum']})
else:
    print("Warning: 'kmeans_cluster' column not found in dataframe")