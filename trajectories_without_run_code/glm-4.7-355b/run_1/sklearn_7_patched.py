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
    try:
        df = pd.read_csv("data/Customers.csv")
    except FileNotFoundError:
        print("Error: Could not find the CSV file.")
    except pd.errors.ParserError:
        print("Error: Could not parse the CSV file.")
    else:
        print("Imported necessary libraries and loaded data successfully.")
        print("Dataset columns:", df.columns.tolist())

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

# Check if the target column exists in the dataframe
target_column = 'Spending Score (1-100)'
if target_column not in df.columns:
    print(f"Error: Column '{target_column}' not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    print("The CSV file appears to be a Git LFS pointer file. The actual data needs to be downloaded.")
else:
    # Split the data into training and testing sets
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)