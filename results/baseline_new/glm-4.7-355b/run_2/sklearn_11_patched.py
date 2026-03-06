# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score



#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv("data/data.csv")


#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# 
# abnormaldata1 = df[df['month'] =='8']
# x = np.array(abnormaldata1.loc[:,"number_people"]).reshape(-1,1)
# y = np.array(abnormaldata1.loc[:,"temperature"]).reshape(-1,1)

# === AFTER (edited) ===
# Check available columns in the dataframe
print("Columns in dataframe:", df.columns.tolist())

# Filter data (assuming the column might be named 'Month' or similar)
# Common month column names could be 'Month', 'month', 'month_name', etc.
# Let's try to find a month-related column
month_col = None
for col in df.columns:
    if 'month' in col.lower():
        month_col = col
        break

if month_col:
    print(f"Using column: {month_col}")
    abnormaldata1 = df[df[month_col] == '8']
else:
    print("No month column found, showing first few rows:")
    print(df.head())
    # If no month column, we'll proceed with all data
    abnormaldata1 = df

x = np.array(abnormaldata1.loc[:,"number_people"]).reshape(-1,1)
y = np.array(abnormaldata1.loc[:,"temperature"]).reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(r2_score(y_test, y_pred))
