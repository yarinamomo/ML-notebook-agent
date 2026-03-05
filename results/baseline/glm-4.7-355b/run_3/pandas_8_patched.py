# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np # linear algebra
import pandas as pd

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df=pd.read_csv("data/2. Cars Data1.csv")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# df["DriveTrain"]=df["DriveTrain"].astype("category")

# === AFTER (edited) ===
# Check available columns and find the DriveTrain column (case-insensitive)
available_cols = df.columns.tolist()
print("Available columns:", available_cols)

# Find the DriveTrain column (might have different case or exact spelling)
drivetrain_col = None
for col in available_cols:
    if col.lower().replace(' ', '').replace('_', '') == 'drivetrain':
        drivetrain_col = col
        break

if drivetrain_col:
    print(f"Using column: '{drivetrain_col}'")
    df[drivetrain_col] = df[drivetrain_col].astype("category")
else:
    print("Warning: 'DriveTrain' column not found in the dataframe")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
df["MSRP"]=pd.to_numeric(df["MSRP"]) 