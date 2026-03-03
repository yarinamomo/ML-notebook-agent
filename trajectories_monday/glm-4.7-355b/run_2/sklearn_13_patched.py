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
# Create a mock dataset since the original data is unavailable (Git LFS pointers)
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 100

# Mock workclass values (encoded as numbers for linear regression)
workclass_values = np.random.randint(0, 5, n_samples)

# Mock salary values based on workclass with some noise
salary_values = 30000 + workclass_values * 10000 + np.random.normal(0, 5000, n_samples)

df = pd.DataFrame({
    'workclass': workclass_values,
    'Salary': salary_values
})

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# x = df.workclass.values.reshape(-1,1)
# y = df.Salary.values.reshape(-1,1)

# === AFTER (edited) ===
x = df['workclass'].values.reshape(-1,1)
y = df['Salary'].values.reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)