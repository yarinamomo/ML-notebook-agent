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
# Strip whitespace from column names and handle potential naming variations
df.columns = df.columns.str.strip()
# Find DriveTrain column (case-insensitive) and convert to category
drivetrain_col = next((col for col in df.columns if col.lower() == 'drivetrain'), None)
if drivetrain_col:
    df[drivetrain_col] = df[drivetrain_col].astype("category")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 4}
df["MSRP"]=pd.to_numeric(df["MSRP"]) 