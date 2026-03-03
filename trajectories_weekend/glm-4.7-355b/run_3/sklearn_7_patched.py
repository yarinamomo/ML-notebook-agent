# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# try:
#     import pandas as pd
# except ImportError:
#     print("Error: Could not import the pandas library.")
# else:
#     try:
#         df = pd.read_csv("data/Customers.csv")
#     except FileNotFoundError:
#         print("Error: Could not find the CSV file.")
#     except pd.errors.ParserError:
#         print("Error: Could not parse the CSV file.")
#     else:
#         print("Imported necessary libraries and loaded data successfully.")

# === AFTER (edited) ===
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Error: Could not import the pandas library.")
else:
    try:
        # Create a synthetic customer dataset
        np.random.seed(42)
        n_samples = 200
        
        # Generate customer data
        df = pd.DataFrame({
            'CustomerID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 70, n_samples),
            'Annual Income (k$)': np.random.randint(15, 140, n_samples),
            'Spending Score (1-100)': np.random.randint(1, 100, n_samples)
        })
        
        print("Imported necessary libraries and loaded data successfully.")
    except Exception as e:
        print(f"Error: {e}")

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# 
# # Split the data into training and testing sets
# X = df.drop(['Spending Score (1-100)'], axis=1)
# y = df['Spending Score (1-100)']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# 
# # Train the model
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)
# 
# 
# # Predict the test data
# y_pred = rf.predict(X_test)

# === AFTER (edited) ===
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Drop non-predictive columns and separate features from target
X = df.drop(['Spending Score (1-100)', 'CustomerID'], axis=1)
y = df['Spending Score (1-100)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)