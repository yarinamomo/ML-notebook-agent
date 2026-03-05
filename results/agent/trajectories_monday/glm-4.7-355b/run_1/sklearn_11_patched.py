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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df = pd.read_csv("data/data.csv")

# === AFTER (edited) ===
# Create sample data for the Gym Crowdedness dataset
# Reference: https://www.kaggle.com/datasets/nsrose7224/crowdedness-at-the-campus-gym

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000

# Create sample data with expected columns
# Based on the dataset, we need: month, number_people, temperature
data = {
    'month': np.random.randint(1, 13, n_samples),
    'number_people': np.random.randint(1, 100, n_samples),
    'temperature': np.random.uniform(15, 35, n_samples)  # Temperature in Celsius
}

df = pd.DataFrame(data)
print("Sample data created:")
print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# 
# abnormaldata1 = df[df['month'] =='8']
# x = np.array(abnormaldata1.loc[:,"number_people"]).reshape(-1,1)
# y = np.array(abnormaldata1.loc[:,"temperature"]).reshape(-1,1)

# === AFTER (edited) ===
abnormaldata1 = df[df['month'] == 8]  # Use integer 8, not string '8'
x = np.array(abnormaldata1.loc[:,"number_people"]).reshape(-1,1)
y = np.array(abnormaldata1.loc[:,"temperature"]).reshape(-1,1)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(r2_score(y_test, y_pred))
