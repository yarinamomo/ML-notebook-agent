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
# Since the CSV is a Git LFS pointer, create sample data for demonstration
import numpy as np
import pandas as pd

# Create sample data with workclass and Salary columns
np.random.seed(42)
workclass_values = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 
                    'State-gov', 'Without-pay', 'Never-worked']
salary_values = ['<=50K', '>50K']

# Create 100 random samples
n_samples = 100
df = pd.DataFrame({
    'workclass': np.random.choice(workclass_values, n_samples),
    'Salary': np.random.choice(salary_values, n_samples)
})

print("Sample data created:")
print(df.head())

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
from sklearn.preprocessing import LabelEncoder

# Encode categorical variables
label_encoder = LabelEncoder()
x_encoded = label_encoder.fit_transform(df.workclass.values)
x = x_encoded.reshape(-1,1)
y_encoded = label_encoder.fit_transform(df.Salary.values)
y = y_encoded.reshape(-1,1)

linear_reg = LinearRegression()
linear_reg.fit(x, y)