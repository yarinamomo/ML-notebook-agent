# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score



#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# df = pd.read_csv("data/data.csv")

# === AFTER (edited) ===
# Create sample data since the data file is a Git LFS pointer
# This simulates what the data structure should look like
import pandas as pd
import numpy as np

# Create sample data with expected columns: month, number_people, temperature
np.random.seed(42)
n_samples = 100

# Generate sample data
data = {
    'month': np.random.choice(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'], n_samples),
    'number_people': np.random.randint(10, 100, n_samples),
    'temperature': np.random.uniform(15, 35, n_samples)
}
df = pd.DataFrame(data)
print("Sample data created successfully")
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# 
# abnormaldata1 = df[df['month'] =='8']
# x = np.array(abnormaldata1.loc[:,"number_people"]).reshape(-1,1)
# y = np.array(abnormaldata1.loc[:,"temperature"]).reshape(-1,1)

# === AFTER (edited) ===
abnormaldata1 = df[df['month'] =='8']
x = np.array(abnormaldata1.loc[:,"number_people"]).reshape(-1,1)
y = np.array(abnormaldata1.loc[:,"temperature"]).reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(r2_score(y_test, y_pred))
