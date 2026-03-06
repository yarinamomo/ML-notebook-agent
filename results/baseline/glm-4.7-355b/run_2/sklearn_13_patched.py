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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# x = df.workclass.values.reshape(-1,1)
# y = df.Salary.values.reshape(-1,1)

# === AFTER (edited) ===
# Check available columns to find the correct names
print("Available columns:", df.columns.tolist())

# Use the actual column names (adjust based on what's in the CSV)
# If the columns have different names, we'll adapt
x = df.iloc[:, 0].values.reshape(-1,1)  # First column as feature
y = df.iloc[:, -1].values.reshape(-1,1)  # Last column as target

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)