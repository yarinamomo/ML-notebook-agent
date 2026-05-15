# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import os, datetime, sys, random, time
 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
 
plt.style.use('fivethirtyeight')
%matplotlib inline
 
from scipy import stats, special
import shap                # 
 
import warnings
warnings.filterwarnings('ignore')
 
train_data=pd.read_csv("data/cs-training_synthetic.csv",encoding="utf-8")
test_data=pd.read_csv("data/cs-test_synthetic.csv",encoding="utf-8")
 
print(train_data.head())
print(test_data.head())

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# remove id 
dev_train=train_data.drop("Unnamed: 0",axis=1)
# 测试集也做同样操作
print(test_data.info())
dev_test=test_data.drop("Unnamed: 0",axis=1)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# import seaborn as sns
# # 检查数据正负样本是否平衡
# fig,axes=plt.subplots(1,2,figsize=(12,6))
# # pandas自带绘图
# dev_train['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct="%1.1f%%",ax=axes[0])
# axes[0].set_title("SeriousDlqin2yrs")
# sns.countplot("SeriousDlqin2yrs",data=dev_train,ax=axes[1])
# axes[1].set_title("SeriousDlqin2yrs")
# plt.show()

# === AFTER (edited) ===
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

dev_train['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0, 0.1], autopct="%1.1f%%", ax=axes[0])
axes[0].set_title("SeriousDlqin2yrs")
# Use keyword arguments to be compatible with newer seaborn versions
sns.countplot(x="SeriousDlqin2yrs", data=dev_train, ax=axes[1])
axes[1].set_title("SeriousDlqin2yrs")
plt.show()