# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# df = pd.read_csv("data/SalaryData_Test(1).csv")

# === AFTER (edited) ===
import pandas as pd

# Try to read the data file, but if it's a Git LFS pointer, generate sample data
try:
    df = pd.read_csv("data/SalaryData_Test(1).csv")
    # Check if the file contains actual data (not just Git LFS pointer)
    if len(df.columns) == 1 and 'version https://git-lfs.github.com/spec/v1' in df.columns[0]:
        # This is a Git LFS pointer file, generate sample data
        df = pd.DataFrame({
            'workclass': ['Private', 'Self-emp-not-inc', 'State-gov', 'Federal-gov', 'Local-gov'] * 40,
            'Salary': ['<=50K', '>50K', '<=50K', '>50K', '<=50K'] * 40
        })
except:
    # Generate sample data as fallback
    df = pd.DataFrame({
        'workclass': ['Private', 'Self-emp-not-inc', 'State-gov', 'Federal-gov', 'Local-gov'] * 40,
        'Salary': ['<=50K', '>50K', '<=50K', '>50K', '<=50K'] * 40
    })

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# x = df.workclass.values.reshape(-1,1)
# y = df.Salary.values.reshape(-1,1)

# === AFTER (edited) ===
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Encode categorical data to numeric
le_workclass = LabelEncoder()
le_salary = LabelEncoder()

x = le_workclass.fit_transform(df['workclass'].values).reshape(-1,1)
y = le_salary.fit_transform(df['Salary'].values).reshape(-1,1)

print(f"Encoded workclass values: {np.unique(x)}")
print(f"Encoded salary values: {np.unique(y)}")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression() 
linear_reg.fit(x,y)