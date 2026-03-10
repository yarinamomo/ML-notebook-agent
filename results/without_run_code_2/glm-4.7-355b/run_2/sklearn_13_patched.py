# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df = pd.read_csv("data/SalaryData_Test(1).csv")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# x = df.workclass.values.reshape(-1,1)
# y = df.Salary.values.reshape(-1,1)

# === AFTER (edited) ===
from sklearn.preprocessing import LabelEncoder

# Encode categorical features to numeric
le_x = LabelEncoder()
le_y = LabelEncoder()

x = le_x.fit_transform(df.workclass.values).reshape(-1,1)
y = le_y.fit_transform(df.Salary.values).reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)