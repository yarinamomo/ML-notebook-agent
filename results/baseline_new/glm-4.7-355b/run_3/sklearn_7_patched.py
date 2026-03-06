# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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


# Find the correct column name for the target variable
target_col = None
for col in df.columns:
    if 'spending' in col.lower() and 'score' in col.lower():
        target_col = col
        break

if target_col is None:
    # Try another pattern if the above doesn't match
    for col in df.columns:
        if 'score' in col.lower():
            target_col = col
            break

if target_col is None:
    # Fall back to taking the last column if not found
    target_col = df.columns[-1]

X = df.drop([target_col], axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)



y_pred = rf.predict(X_test)