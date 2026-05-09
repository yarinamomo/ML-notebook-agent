# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df_ = pd.read_csv("data/dataset.csv", compression="gzip")
df = df_.copy()
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
cat_cols, num_cols, num_but_cat = grab_col_names(df)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
df= df.iloc[:,1:]

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
cat_cols, num_cols, num_but_cat = grab_col_names(df)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
df2=df.copy()

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
sc = MinMaxScaler((0, 1))
df2[num_cols] = sc.fit_transform(df2[num_cols])

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
kmeans = KMeans(n_clusters=30, n_init="auto").fit(df2[["TotalTime","Calories","SugarContent"]])

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
clusters_kmeans = kmeans.labels_
clusters_kmeans

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
df["kmeans_cluster"] = clusters_kmeans
df["kmeans_cluster"]= df["kmeans_cluster"] + 1
df.head()

#%%
# --- [CELL 15]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
# === BEFORE (original) ===
# df.groupby('kmeans_cluster').agg({1: ['count','mean', 'median', 'sum'],
#                                     2: ['count','mean', 'median', 'sum'],
#                                     3: ['count','mean', 'median', 'sum'],
#                                     4: ['count','mean','median', 'sum']})

# === AFTER (edited) ===
df.groupby('kmeans_cluster').agg({
    'TotalTime': ['count', 'mean', 'median', 'sum'],
    'Calories': ['count', 'mean', 'median', 'sum'],
    'SugarContent': ['count', 'mean', 'median', 'sum']
})