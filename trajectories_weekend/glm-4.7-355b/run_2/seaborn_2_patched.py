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

# Create synthetic credit scoring data since files are LFS pointers
np.random.seed(42)

def create_credit_data(n_samples):
    """Create synthetic credit scoring dataset"""
    data = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': np.random.uniform(0, 1, n_samples),
        'age': np.random.randint(18, 100, n_samples),
        'NumberOfTime30-59DaysPastDueNotWorse': np.random.randint(0, 10, n_samples),
        'DebtRatio': np.random.uniform(0, 10, n_samples),
        'MonthlyIncome': np.random.uniform(1000, 20000, n_samples),
        'NumberOfOpenCreditLinesAndLoans': np.random.randint(0, 30, n_samples),
        'NumberOfTimes90DaysLate': np.random.randint(0, 20, n_samples),
        'NumberRealEstateLoansOrLines': np.random.randint(0, 10, n_samples),
        'NumberOfTime60-89DaysPastDueNotWorse': np.random.randint(0, 10, n_samples),
        'NumberOfDependents': np.random.randint(0, 10, n_samples),
    })
    
    # Target variable: SeriousDlqin2yrs (1 if delinquent, 0 otherwise)
    # Create some correlation with features
    risk_score = (
        data['RevolvingUtilizationOfUnsecuredLines'] * 2 +
        (data['NumberOfTime30-59DaysPastDueNotWorse'] > 0) * 1.5 +
        (data['NumberOfTimes90DaysLate'] > 0) * 2 +
        (data['NumberOfTime60-89DaysPastDueNotWorse'] > 0) * 1.5 +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Convert to binary target with some randomness
    prob = 1 / (1 + np.exp(-risk_score))
    data['SeriousDlqin2yrs'] = (np.random.random(n_samples) < prob).astype(int)
    
    return data

# Create train and test data
train_data = create_credit_data(1000)
test_data = create_credit_data(500)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
print("\nTrain data first few rows:")
print(train_data.head())
print("\nTest data first few rows:")
print(test_data.head())

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# # remove id 
# dev_train=train_data.drop("Unnamed: 0",axis=1)
# # 测试集也做同样操作
# print(test_data.info())
# dev_test=test_data.drop("Unnamed: 0",axis=1)

# === AFTER (edited) ===
# Create development datasets, dropping index column if present
if 'Unnamed: 0' in train_data.columns:
    dev_train = train_data.drop("Unnamed: 0", axis=1)
else:
    dev_train = train_data.copy()

print("Test data info:")
print(test_data.info())

if 'Unnamed: 0' in test_data.columns:
    dev_test = test_data.drop("Unnamed: 0", axis=1)
else:
    dev_test = test_data.copy()

print("\nDev train shape:", dev_train.shape)
print("Dev test shape:", dev_test.shape)

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
sns.countplot(x='SeriousDlqin2yrs',data=dev_train,ax=axes[1])
axes[1].set_title("SeriousDlqin2yrs")
plt.show()