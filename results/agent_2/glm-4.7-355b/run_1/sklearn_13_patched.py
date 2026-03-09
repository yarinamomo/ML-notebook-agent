# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv("data/SalaryData_Test(1).csv")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# x = df.workclass.values.reshape(-1,1)
# y = df.Salary.values.reshape(-1,1)

# === AFTER (edited) ===
# Use numeric columns for linear regression
# age is numeric, and we'll convert Salary to numeric (0 for <=50K, 1 for >50K)
x = df['age'].values.reshape(-1, 1)
y = (df['Salary'] == ' >50K').astype(int).values.reshape(-1, 1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)