# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")



#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# df = pd.read_csv('data/creditcard.csv')
# df.head()

# === AFTER (edited) ===
# Create synthetic credit card data mimicking the typical structure
# Since the actual data file is a Git LFS pointer, we generate synthetic data
np.random.seed(42)
n_samples = 1000
n_fraud = 20  # 2% fraud rate

# Generate synthetic features
data = {
    'Time': np.random.uniform(0, 172800, n_samples),  # 48 hours in seconds
    'Amount': np.exp(np.random.normal(3, 1.5, n_samples)),  # Log-normal distribution
}

# Generate V1-V28 features (PCA-transformed features typical in credit card datasets)
for i in range(1, 29):
    data[f'V{i}'] = np.random.normal(0, 1, n_samples)

# Create Class column (0 for normal, 1 for fraud)
data['Class'] = np.array([0] * (n_samples - n_fraud) + [1] * n_fraud)
np.random.shuffle(data['Class'])

df = pd.DataFrame(data)
df.head()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
data_df = df.copy()
data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
data_df_1 = pd.DataFrame(tmp)
data_df_1.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
data_df_1.head()

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
# === BEFORE (original) ===
# var = data_df.columns.values
# 
# i = 0
# t0 = df.loc[df['Class'] == 0]
# t1 = df.loc[df['Class'] == 1]
# 
# sns.set_style('whitegrid')
# plt.figure()
# fig, ax = plt.subplots(8,4,figsize=(16,28))
# 
# for feature in var:
#     i += 1
#     plt.subplot(8,4,i)
#     sns.kdeplot(t0[feature], bw=0.5,label="Class = 0");
#     sns.kdeplot(t1[feature], bw=0.5,label="Class = 1");
#     plt.xlabel(feature, fontsize=12)
#     locs, labels = plt.xticks()
#     plt.tick_params(axis='both', which='major', labelsize=12)
# plt.show();

# === AFTER (edited) ===
var = df.columns.values

i = 0
t0 = df.loc[df['Class'] == 0]
t1 = df.loc[df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw=0.5,label="Class = 0");
    sns.kdeplot(t1[feature], bw=0.5,label="Class = 1");
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();