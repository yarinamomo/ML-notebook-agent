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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
x = df.workclass.values.reshape(-1,1)
y = df.Salary.values.reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# from sklearn.linear_model import LinearRegression
# 
# linear_reg = LinearRegression() 
# linear_reg.fit(x,y)

# === AFTER (edited) ===
from sklearn.linear_model import LinearRegression

# Encode categorical feature and target to numeric values
x_encoded = pd.Categorical(x.ravel()).codes.reshape(-1, 1)
y_encoded = pd.Categorical(y.ravel()).codes

linear_reg = LinearRegression()
linear_reg.fit(x_encoded, y_encoded)