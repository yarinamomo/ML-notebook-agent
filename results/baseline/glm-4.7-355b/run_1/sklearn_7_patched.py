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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 2}
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
import pandas as pd

# Print available columns
print("Available columns in df:", df.columns.tolist())

# Try to identify the target column (spending/score related)
target_col = None
for col in df.columns:
    if 'Score' in col or 'Spending' in col:
        target_col = col
        break

if target_col is None:
    raise KeyError(f"No spending/score column found. Available columns: {df.columns.tolist()}")

print(f"Using target column: '{target_col}'")

# Drop the target column and non-numeric columns like ID and Name
# Get numeric features only (exclude the target)
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
if target_col in numeric_cols:
    numeric_cols.remove(target_col)

X = df[numeric_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)