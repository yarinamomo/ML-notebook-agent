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
# Create synthetic data matching the expected structure
# The notebook expects columns: month, number_people, temperature
data = {
    'month': ['1', '1', '2', '2', '3', '3', '4', '4', '5', '5',
              '6', '6', '7', '7', '8', '8', '8', '8', '9', '9',
              '10', '10', '11', '11', '12', '12'],
    'number_people': [10, 15, 20, 25, 30, 35, 40, 45, 50, 55,
                      60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
                      110, 115, 120, 125, 130, 135],
    'temperature': [15.0, 16.0, 18.0, 19.0, 22.0, 24.0, 26.0, 28.0, 
                    30.0, 32.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 
                    40.0, 41.0, 38.0, 37.0, 35.0, 33.0, 30.0, 28.0, 
                    25.0, 22.0]
}

df = pd.DataFrame(data)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}

abnormaldata1 = df[df['month'] =='8']
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
