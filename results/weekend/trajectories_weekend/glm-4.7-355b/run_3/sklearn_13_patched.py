# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df = pd.read_csv("data/SalaryData_Test(1).csv")

# === AFTER (edited) ===
# Create synthetic data since actual CSV files are Git LFS pointers
np.random.seed(42)
n_samples = 100

# Create a numeric feature (e.g., years of experience) and salary target
x_data = np.random.uniform(0, 40, n_samples)  # years of experience
y_data = 30000 + 2500 * x_data + np.random.normal(0, 5000, n_samples)  # salary with noise

df = pd.DataFrame({'years_experience': x_data, 'Salary': y_data})
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# x = df.workclass.values.reshape(-1,1)
# y = df.Salary.values.reshape(-1,1)

# === AFTER (edited) ===
x = df.years_experience.values.reshape(-1,1)
y = df.Salary.values.reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)