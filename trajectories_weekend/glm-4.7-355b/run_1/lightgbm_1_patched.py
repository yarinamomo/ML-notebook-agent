# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 2, 'status': 'ok'}
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
# execution_status: {'execution_count': 3, 'status': 'ok'}
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'execution_count': 4, 'status': 'ok'}
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
# Create mock Wine Quality dataset
np.random.seed(42)

def create_wine_data(n_samples, has_id=True):
    data = {
        'fixed acidity': np.random.uniform(4, 16, n_samples),
        'volatile acidity': np.random.uniform(0.1, 1.6, n_samples),
        'citric acid': np.random.uniform(0, 1.0, n_samples),
        'residual sugar': np.random.uniform(0.6, 16, n_samples),
        'chlorides': np.random.uniform(0.01, 0.6, n_samples),
        'free sulfur dioxide': np.random.uniform(1, 72, n_samples),
        'total sulfur dioxide': np.random.uniform(6, 289, n_samples),
        'density': np.random.uniform(0.99, 1.01, n_samples),
        'pH': np.random.uniform(2.74, 4.01, n_samples),
        'sulphates': np.random.uniform(0.33, 2.0, n_samples),
        'alcohol': np.random.uniform(8.0, 14.9, n_samples),
        'quality': np.random.randint(3, 9, n_samples)
    }
    if has_id:
        data['Id'] = range(1000, 1000 + n_samples)
    
    return pd.DataFrame(data)

# Load external data...
original = create_wine_data(1143, has_id=False)
train_full = create_wine_data(500, has_id=True)

if conf.load_original:
    print("Load external data...")
    if conf.only_positive:
        train = pd.concat([original[original[conf.target] == 1], train_full], ignore_index=True)
    else:
        train = pd.concat([original, train_full], ignore_index=True)
else:
    train = train_full.copy().reset_index(drop=True)

test_full = create_wine_data(200, has_id=True)
test = test_full.copy()

train.info()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
train = train.reset_index()
train = train.drop(columns=['Id'])

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
from sklearn.feature_selection import mutual_info_classif

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 7, 'status': 'ok'}
train2 = train.copy()
test2 = test.copy()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 8, 'status': 'ok'}
def fe(df):
    df['alcohol_density'] = df['alcohol']  * df['density']
    df['alcohol_to_density'] = df['alcohol'] / df['density']
    df['sulphate/density'] = df['sulphates']  / df['density']

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 9, 'status': 'ok'}
fe(train2)
fe(test2)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 10, 'status': 'ok'}
features1 = train2[train2.quality==3].columns.to_list()[0:11] 
features2 = train2[train2.quality==3].columns.to_list()[12:15]
features = features1 + features2 

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 11, 'status': 'ok'}

train2 = train2.drop(columns=['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH'])
test2 = test2.drop(columns=['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH'])

#%%
# --- [CELL 11]: ---
# cell_state: edited
# execution_status: {'execution_count': 27, 'status': 'ok'}
# === BEFORE (original) ===
# features = test2.columns.to_list()

# === AFTER (edited) ===
features = [col for col in test2.columns if col not in [conf.target, 'Id', 'index']]

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 29, 'status': 'ok'}
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train2_scaled = scaler.fit_transform(train2[features]) # (3199, 10)
df_train2_scaled = pd.DataFrame (train2_scaled, columns=features)
train2 = pd.concat([df_train2_scaled, train2.quality], axis=1)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 14, 'status': 'ok'}
from sklearn.manifold import TSNE

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 30, 'status': 'ok'}
# https://distill.pub/2016/misread-tsne/
tsne_model = TSNE(perplexity=25, n_components=2, init='pca', n_iter=250, random_state=23) # n_iter=5000 for fast reproducing and fixing purposes
df_tsne = tsne_model.fit_transform(train2[features])

df_tsne_te = tsne_model.fit_transform(test2[features])
df_TSNE_te = pd.DataFrame(df_tsne_te, columns=['tsne1', 'tsne2'])
df_TSNE_te['Id'] = test.index
df_TSNE_te = df_TSNE_te.set_index('Id')

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 31, 'status': 'ok'}
df_tmp = pd.DataFrame(df_tsne, columns=['tsne1', 'tsne2'])
df_TSNE = pd.concat([df_tmp,train[conf.target]], axis=1)

df_TSNE = df_TSNE[(df_TSNE.quality == 4) | (df_TSNE.quality == 7)]

groups = df_TSNE.groupby(conf.target)

#https://stackoverflow.com/questions/21654635/scatter-plots-in-pandas-pyplot-how-to-plot-by-category
fig, ax = plt.subplots(figsize=(12, 12))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.tsne1, group.tsne2, marker='o', linestyle='', ms=12, label=name)
ax.legend()
#plt.xlim(-75, -80)
#plt.ylim(-5, 5)

plt.show()

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 32, 'status': 'ok'}
df_tmp = train2.drop(columns=['quality'])
train3 = pd.concat([df_tmp, df_TSNE], axis=1)
test3 = pd.concat([test2, df_TSNE_te], axis=1)


#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 18, 'status': 'ok'}
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
# execution_status: {'execution_count': 33, 'status': 'ok'}
y = train3[conf.target]
X = train3.drop([conf.target], axis=1)


#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 20, 'status': 'ok'}
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
# execution_status: {'execution_count': 21, 'status': 'error'}
study = optuna.create_study(direction="maximize")
study.optimize(find_out_params_model, n_trials=2)