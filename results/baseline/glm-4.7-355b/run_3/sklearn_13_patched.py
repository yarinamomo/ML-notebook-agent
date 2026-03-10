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

# Encode the categorical workclass column to numerical values
label_encoder = LabelEncoder()
x_encoded = label_encoder.fit_transform(df.workclass.values)
x = x_encoded.reshape(-1,1)
y = df.Salary.values.reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)