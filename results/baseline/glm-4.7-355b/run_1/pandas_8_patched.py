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
# First check what columns are available and find the DriveTrain/Drivetrain column
print("Available columns:", df.columns.tolist())

# Try to find the DriveTrain column (handle case variations)
drivetrain_col = None
for col in df.columns:
    # Remove spaces and check both cases
    col_name = col.replace(' ', '')
    if col_name.lower() == 'drivetrain':
        drivetrain_col = col
        break

if drivetrain_col:
    print(f"Found 'DriveTrain' column as: '{drivetrain_col}'")
    df[drivetrain_col] = df[drivetrain_col].astype('category')
else:
    print("Warning: DriveTrain column not found in dataframe")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df["MSRP"]=pd.to_numeric(df["MSRP"]) 

# === AFTER (edited) ===
# Convert MSRP to numeric, handling any potential column name variations
# Try to find the MSRP column
msrp_col = None
for col in df.columns:
    if col.replace(' ', '').upper() == 'MSRP':
        msrp_col = col
        break

if msrp_col:
    print(f"Found 'MSRP' column as: '{msrp_col}'")
    df[msrp_col] = pd.to_numeric(df[msrp_col], errors='coerce')
else:
    print("Warning: MSRP column not found in dataframe")