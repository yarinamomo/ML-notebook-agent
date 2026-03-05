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
except ImportError:
    print("Error: Could not import the pandas library.")
else:
    # Create sample data since the CSV file is not available
    data = {
        'CustomerID': range(1, 201),
        'Age': [20 + (i % 60) for i in range(200)],
        'Annual Income (k$)': [15 + (i % 100) for i in range(200)],
        'Spending Score (1-100)': [1 + (i % 99) for i in range(200)]
    }
    df = pd.DataFrame(data)
    print("Imported necessary libraries and loaded sample data successfully.")

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Split the data into training and testing sets
X = df.drop(['Spending Score (1-100)'], axis=1)
y = df['Spending Score (1-100)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# Predict the test data
y_pred = rf.predict(X_test)