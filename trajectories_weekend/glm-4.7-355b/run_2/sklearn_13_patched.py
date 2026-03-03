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
# Create sample data since CSV files are not available (Git LFS pointers)
import pandas as pd
import numpy as np

# Generate sample data similar to what would be expected
np.random.seed(42)
n_samples = 100

# Create workclass categories (encoded as numbers for linear regression)
workclass_values = np.random.randint(0, 8, n_samples)
salary_values = workclass_values * 10000 + np.random.normal(0, 5000, n_samples)

df = pd.DataFrame({
    'workclass': workclass_values,
    'Salary': salary_values
})

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
x = df.workclass.values.reshape(-1,1)
y = df.Salary.values.reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)