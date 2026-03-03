# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
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


#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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

# Try to find the spending score column
if 'Spending Score (1-100)' in df.columns:
    target_column = 'Spending Score (1-100)'
elif 'SpendingScore' in df.columns:
    target_column = 'SpendingScore'
elif 'Spending_Score' in df.columns:
    target_column = 'Spending_Score'
elif 'Spending Score' in df.columns:
    target_column = 'Spending Score'
else:
    # Use the last numeric column as target if none match
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    if len(numeric_columns) > 0:
        target_column = numeric_columns[-1]
    else:
        raise ValueError(f"No suitable target column found. Available columns: {df.columns.tolist()}")

X = df.drop([target_column], axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)