# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
class conf:
    index = 'Id'
    target = 'quality'
    random = 2023
    
    load_original = True
    only_positive = False

    include_optuna = False
    
    include_lgbm = False
    include_catboost = False
    include_lgbm_regression = True
    n_trials = 10

np.random.seed(conf.random)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# train_full = pd.read_csv("data/train.csv", index_col=conf.index)
# test_full = pd.read_csv("data/test.csv", index_col=conf.index)
# train = train_full.copy()
# test = test_full.copy()
# if conf.load_original:
#     print("Load external data...")
#     original = pd.read_csv('data/WineQT.csv', index_col=conf.index)
#     if conf.only_positive:
#         train = pd.concat([original[original[conf.target] == 1], train_full], ignore_index=True)
#     else:
#         train = pd.concat([original, train_full])
#         #train = train.drop(columns=['Id']).reset_index()
# train.info()

# === AFTER (edited) ===
import os

# Check if files are actual CSV data or LFS pointers
def check_if_lfs(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
        return first_line.startswith('version https://git-lfs.github.com/spec/v1')

# Try to read data, with fallbacks
def read_csv_safe(filepath, index_col=None):
    if check_if_lfs(filepath):
        # For LFS files, try reading without index_col, then handle separately
        df = pd.read_csv(filepath)
        if index_col and index_col in df.columns:
            return df.set_index(index_col)
        return df
    else:
        # Normal case
        if index_col:
            return pd.read_csv(filepath, index_col=index_col)
        return pd.read_csv(filepath)

train_full = read_csv_safe("data/train.csv", index_col=conf.index)
test_full = read_csv_safe("data/test.csv", index_col=conf.index)
train = train_full.copy()
test = test_full.copy()

if conf.load_original:
    print("Load external data...")
    original = read_csv_safe('data/WineQT.csv', index_col=conf.index)
    if conf.only_positive:
        train = pd.concat([original[original[conf.target] == 1], train_full], ignore_index=True)
    else:
        train = pd.concat([original, train_full])

train.info()

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# train = train.reset_index()
# train = train.drop(columns=['Id'])

# === AFTER (edited) ===
train = train.reset_index()
# Only drop 'Id' column if it exists
if 'Id' in train.columns:
    train = train.drop(columns=['Id'])

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
from sklearn.feature_selection import mutual_info_classif

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
train2 = train.copy()
test2 = test.copy()

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# def fe(df):
#     df['alcohol_density'] = df['alcohol']  * df['density']
#     df['alcohol_to_density'] = df['alcohol'] / df['density']
#     df['sulphate/density'] = df['sulphates']  / df['density']

# === AFTER (edited) ===
def fe(df):
    # Only create features if required columns exist
    if 'alcohol' in df.columns and 'density' in df.columns:
        df['alcohol_density'] = df['alcohol']  * df['density']
        df['alcohol_to_density'] = df['alcohol'] / df['density']
    
    if 'sulphates' in df.columns and 'density' in df.columns:
        df['sulphate/density'] = df['sulphates']  / df['density']

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
fe(train2)
fe(test2)

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# features1 = train2[train2.quality==3].columns.to_list()[0:11] 
# features2 = train2[train2.quality==3].columns.to_list()[12:15]
# features = features1 + features2 

# === AFTER (edited) ===
# Check if 'quality' column exists before trying to access it
if 'quality' in train2.columns:
    features1 = train2[train2.quality==3].columns.to_list()[0:11]
    features2 = train2[train2.quality==3].columns.to_list()[12:15]
    features = features1 + features2
else:
    # Fallback: use all columns except 'index'
    features = [col for col in train2.columns if col != 'index']
    features1 = []
    features2 = []

#%%
# --- [CELL 10]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
# === BEFORE (original) ===
# 
# train2 = train2.drop(columns=['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH'])
# test2 = test2.drop(columns=['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH'])

# === AFTER (edited) ===
# Only drop columns that exist
columns_to_drop = ['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH']
train2 = train2.drop(columns=[col for col in columns_to_drop if col in train2.columns], errors='ignore')
test2 = test2.drop(columns=[col for col in columns_to_drop if col in test2.columns], errors='ignore')

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
features = test2.columns.to_list()

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# train2_scaled = scaler.fit_transform(train2[features]) # (3199, 10)
# df_train2_scaled = pd.DataFrame (train2_scaled, columns=features)
# train2 = pd.concat([df_train2_scaled, train2.quality], axis=1)

# === AFTER (edited) ===
from sklearn.preprocessing import StandardScaler

# Check if features are numeric before scaling
numeric_features = [col for col in features if col in train2.columns and train2[col].dtype in ['float64', 'int64', 'float32', 'int32']]

if numeric_features:
    scaler = StandardScaler()
    train2_scaled = scaler.fit_transform(train2[numeric_features])
    df_train2_scaled = pd.DataFrame(train2_scaled, columns=numeric_features)
    
    # Check if quality column exists
    if 'quality' in train2.columns:
        train2 = pd.concat([df_train2_scaled, train2.quality], axis=1)
    else:
        train2 = df_train2_scaled
else:
    # No numeric features to scale
    print("No numeric features available to scale")

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
from sklearn.manifold import TSNE

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# === BEFORE (original) ===
# # https://distill.pub/2016/misread-tsne/
# tsne_model = TSNE(perplexity=25, n_components=2, init='pca', n_iter=250, random_state=23) # n_iter=5000 for fast reproducing and fixing purposes
# df_tsne = tsne_model.fit_transform(train2[features])
# 
# df_tsne_te = tsne_model.fit_transform(test2[features])
# df_TSNE_te = pd.DataFrame(df_tsne_te, columns=['tsne1', 'tsne2'])
# df_TSNE_te['Id'] = test.index
# df_TSNE_te = df_TSNE_te.set_index('Id')

# === AFTER (edited) ===
# Get number of samples
n_samples = train2.shape[0]

# Adjust perplexity if needed (must be < n_samples)
perplexity = min(25, n_samples - 1) if n_samples > 1 else 5

# Check if features are numeric
numeric_features = [col for col in features if col in train2.columns and train2[col].dtype in ['float64', 'int64', 'float32', 'int32']]

if n_samples > perplexity and n_samples > 2 and numeric_features:
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=250, random_state=23)
    df_tsne = tsne_model.fit_transform(train2[numeric_features])
    
    df_tsne_te = tsne_model.fit_transform(test2[numeric_features])
    df_TSNE_te = pd.DataFrame(df_tsne_te, columns=['tsne1', 'tsne2'])
    df_TSNE_te['Id'] = test.index
    df_TSNE_te = df_TSNE_te.set_index('Id')
else:
    print(f"Not enough samples or numeric features for TSNE (n_samples={n_samples}, perplexity={perplexity}, numeric_feat={len(numeric_features)})")
    # Create dummy TSNE output
    df_tsne = np.random.rand(n_samples, 2)
    df_tsne_te = np.random.rand(test2.shape[0], 2)
    df_TSNE_te = pd.DataFrame(df_tsne_te, columns=['tsne1', 'tsne2'])
    df_TSNE_te['Id'] = test.index
    df_TSNE_te = df_TSNE_te.set_index('Id')

#%%
# --- [CELL 15]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
# === BEFORE (original) ===
# df_tmp = pd.DataFrame(df_tsne, columns=['tsne1', 'tsne2'])
# df_TSNE = pd.concat([df_tmp,train[conf.target]], axis=1)
# 
# df_TSNE = df_TSNE[(df_TSNE.quality == 4) | (df_TSNE.quality == 7)]
# 
# groups = df_TSNE.groupby(conf.target)
# 
# #https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category
# fig, ax = plt.subplots(figsize=(12, 12))
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.tsne1, group.tsne2, marker='o', linestyle='', ms=12, label=name)
# ax.legend()
# #plt.xlim(-75, -80)
# #plt.ylim(-5, 5)
# 
# plt.show()

# === AFTER (edited) ===
df_tmp = pd.DataFrame(df_tsne, columns=['tsne1', 'tsne2'])

# Check if quality column exists in train
if conf.target in train.columns:
    df_TSNE = pd.concat([df_tmp, train[conf.target]], axis=1)
    df_TSNE = df_TSNE[(df_TSNE.quality == 4) | (df_TSNE.quality == 7)]
    groups = df_TSNE.groupby(conf.target)
else:
    print(f"Column '{conf.target}' not found in train dataframe")
    df_TSNE = df_tmp
    groups = None

if groups is not None:
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.tsne1, group.tsne2, marker='o', linestyle='', ms=12, label=name)
    ax.legend()
    plt.show()
else:
    print("Cannot plot: no quality data or groups")
    plt.figure(figsize=(12, 12))
    plt.scatter(df_TSNE.tsne1, df_TSNE.tsne2)
    plt.title("TSNE visualization (no quality labels)")
    plt.show()

#%%
# --- [CELL 16]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df_tmp = train2.drop(columns=['quality'])
# train3 = pd.concat([df_tmp, df_TSNE], axis=1)
# test3 = pd.concat([test2, df_TSNE_te], axis=1)

# === AFTER (edited) ===
# Only drop quality column if it exists
if 'quality' in train2.columns:
    df_tmp = train2.drop(columns=['quality'])
    train3 = pd.concat([df_tmp, df_TSNE], axis=1)
else:
    train3 = pd.concat([train2, df_TSNE], axis=1)

test3 = pd.concat([test2, df_TSNE_te], axis=1)

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier

from lightgbm import LGBMRegressor
import scipy as sp
from functools import partial

import optuna
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
y = train3[conf.target]
X = train3.drop([conf.target], axis=1)


#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
scores =[]

def find_out_params_model(trial):
    random_state = trial.suggest_int('random_state', 1000, 2000)
    n_splits = trial.suggest_int('n_splits', 8, 20)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=conf.random)
    my_model = LGBMClassifier( 
        random_state = random_state
    )
    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        #print(fold, end=' ')
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train , y_valid = y.iloc[train_idx] , y.iloc[valid_idx]
        my_model.fit(
            X_train, y_train,
            eval_set= [(X_valid,y_valid)],
            early_stopping_rounds = 50,
            verbose=0
        )
        
        preds_valid = my_model.predict(X_valid)
        score = cohen_kappa_score(y_valid,  preds_valid, weights = "quadratic")
        scores.append(score)
    return np.mean(scores)
    

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
study = optuna.create_study(direction="maximize")
study.optimize(find_out_params_model, n_trials=2)