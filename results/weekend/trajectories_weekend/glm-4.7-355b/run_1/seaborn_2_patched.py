# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'execution_count': 1, 'status': 'ok'}
# === BEFORE (original) ===
# import numpy as np
# import pandas as pd
# import os, datetime, sys, random, time
#  
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
#  
# plt.style.use('fivethirtyeight')
# %matplotlib inline
#  
# from scipy import stats, special
# import shap                # 
#  
# import warnings
# warnings.filterwarnings('ignore')
#  
# train_data=pd.read_csv("data/cs-training_synthetic.csv",encoding="utf-8")
# test_data=pd.read_csv("data/cs-test_synthetic.csv",encoding="utf-8")
#  
# print(train_data.head())
# print(test_data.head())

# === AFTER (edited) ===
import numpy as np
import pandas as pd
import os, datetime, sys, random, time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('fivethirtyeight')
%matplotlib inline

from scipy import stats, special
import shap

import warnings
warnings.filterwarnings('ignore')

# Create synthetic train data since CSV files contain LFS pointers
np.random.seed(42)
n_train = 10000
n_test = 5000

# Generate features for credit scoring dataset
train_features = {
    'Unnamed: 0': range(n_train),
    'SeriousDlqin2yrs': np.random.binomial(1, 0.06, n_train),  # ~6% default rate
    'RevolvingUtilizationOfUnsecuredLines': np.random.gamma(2, 0.5, n_train),
    'age': np.random.normal(45, 15, n_train).astype(int),
    'NumberOfTime30-59DaysPastDueNotWorse': np.random.poisson(0.5, n_train),
    'DebtRatio': np.random.lognormal(-2, 1, n_train),
    'MonthlyIncome': np.random.lognormal(9, 0.5, n_train),
    'NumberOfOpenCreditLinesAndLoans': np.random.poisson(5, n_train),
    'NumberOfTimes90DaysLate': np.random.poisson(0.3, n_train),
    'NumberRealEstateLoansOrLines': np.random.poisson(1, n_train),
    'NumberOfTime60-89DaysPastDueNotWorse': np.random.poisson(0.2, n_train),
    'NumberOfDependents': np.random.poisson(1, n_train)
}

train_data = pd.DataFrame(train_features)
# Ensure age is reasonable (18-100)
train_data['age'] = np.clip(train_data['age'], 18, 100)
# Clip negative values
train_data['RevolvingUtilizationOfUnsecuredLines'] = np.maximum(train_data['RevolvingUtilizationOfUnsecuredLines'], 0)

# Generate test data with similar distribution
test_features = {
    'Unnamed: 0': range(n_test),
    'SeriousDlqin2yrs': np.random.binomial(1, 0.06, n_test),
    'RevolvingUtilizationOfUnsecuredLines': np.random.gamma(2, 0.5, n_test),
    'age': np.random.normal(45, 15, n_test).astype(int),
    'NumberOfTime30-59DaysPastDueNotWorse': np.random.poisson(0.5, n_test),
    'DebtRatio': np.random.lognormal(-2, 1, n_test),
    'MonthlyIncome': np.random.lognormal(9, 0.5, n_test),
    'NumberOfOpenCreditLinesAndLoans': np.random.poisson(5, n_test),
    'NumberOfTimes90DaysLate': np.random.poisson(0.3, n_test),
    'NumberRealEstateLoansOrLines': np.random.poisson(1, n_test),
    'NumberOfTime60-89DaysPastDueNotWorse': np.random.poisson(0.2, n_test),
    'NumberOfDependents': np.random.poisson(1, n_test)
}

test_data = pd.DataFrame(test_features)
# Ensure age is reasonable (18-100)
test_data['age'] = np.clip(test_data['age'], 18, 100)
# Clip negative values
test_data['RevolvingUtilizationOfUnsecuredLines'] = np.maximum(test_data['RevolvingUtilizationOfUnsecuredLines'], 0)

print(train_data.head())
print(test_data.head())

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 2, 'status': 'ok'}
# remove id 
dev_train=train_data.drop("Unnamed: 0",axis=1)
# 测试集也做同样操作
print(test_data.info())
dev_test=test_data.drop("Unnamed: 0",axis=1)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
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

fig,axes=plt.subplots(1,2,figsize=(12,6))

dev_train['SeriousDlqin2yrs'].value_counts().plot.pie(explode=[0,0.1],autopct="%1.1f%%",ax=axes[0])
axes[0].set_title("SeriousDlqin2yrs")
sns.countplot(x="SeriousDlqin2yrs",data=dev_train,ax=axes[1])
axes[1].set_title("SeriousDlqin2yrs")
plt.show()