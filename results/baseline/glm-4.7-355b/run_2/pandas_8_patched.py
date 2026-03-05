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
# Check available columns and find the correct DriveTrain column name
print("Available columns:", df.columns.tolist())

# Try to find a column that matches 'DriveTrain' (case-insensitive)
drivetrain_col = None
for col in df.columns:
    if 'drive' in col.lower() and 'train' in col.lower():
        drivetrain_col = col
        break
    elif col.strip().lower() == 'drivetrain':
        drivetrain_col = col
        break

if drivetrain_col:
    df[drivetrain_col]=df[drivetrain_col].astype("category")
    print(f"Successfully converted '{drivetrain_col}' to category type")
else:
    print("Warning: Could not find a DriveTrain column in the dataframe")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
df["MSRP"]=pd.to_numeric(df["MSRP"]) 