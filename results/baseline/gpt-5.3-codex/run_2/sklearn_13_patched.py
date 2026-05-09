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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# from sklearn.linear_model import LinearRegression
# 
# linear_reg = LinearRegression() 
# linear_reg.fit(x,y)

# === AFTER (edited) ===
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Encode categorical feature before fitting linear regression
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), [0])],
    remainder='drop'
)

linear_reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

linear_reg.fit(x, y)