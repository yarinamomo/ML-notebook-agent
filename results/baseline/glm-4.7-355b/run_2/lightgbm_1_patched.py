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
train_full = pd.read_csv("data/train.csv")
test_full = pd.read_csv("data/test.csv")
train = train_full.copy()
test = test_full.copy()
if conf.load_original:
    print("Load external data...")
    original = pd.read_csv('data/WineQT.csv')
    if conf.only_positive:
        train = pd.concat([original[original[conf.target] == 1], train_full], ignore_index=True)
    else:
        train = pd.concat([original, train_full])

train.info()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
train = train.reset_index()
train = train.drop(columns=['Id'])

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.feature_selection import mutual_info_classif

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train2 = train.copy()
test2 = test.copy()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
def fe(df):
    df['alcohol_density'] = df['alcohol']  * df['density']
    df['alcohol_to_density'] = df['alcohol'] / df['density']
    df['sulphate/density'] = df['sulphates']  / df['density']

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
fe(train2)
fe(test2)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
features1 = train2[train2.quality==3].columns.to_list()[0:11] 
features2 = train2[train2.quality==3].columns.to_list()[12:15]
features = features1 + features2 

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}

train2 = train2.drop(columns=['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH'])
test2 = test2.drop(columns=['residual sugar', 'chlorides', 'free sulfur dioxide', 'pH'])

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
features = test2.columns.to_list()

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train2_scaled = scaler.fit_transform(train2[features]) # (3199, 10)
df_train2_scaled = pd.DataFrame (train2_scaled, columns=features)
train2 = pd.concat([df_train2_scaled, train2.quality], axis=1)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.manifold import TSNE

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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
# execution_status: {'status': 'not run'}
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
# execution_status: {'status': 'not run'}
df_tmp = train2.drop(columns=['quality'])
train3 = pd.concat([df_tmp, df_TSNE], axis=1)
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