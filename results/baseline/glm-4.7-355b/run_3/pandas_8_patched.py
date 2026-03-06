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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# df["DriveTrain"]=df["DriveTrain"].astype("category")

# === AFTER (edited) ===
# Try to find the DriveTrain column with case-insensitive matching
dt_column = None
for col in df.columns:
    if col.lower() == 'drivetrain':
        dt_column = col
        break

if dt_column:
    df[dt_column] = df[dt_column].astype("category")
else:
    # Print available columns for debugging
    print("Available columns:", df.columns.tolist())
    raise KeyError("Could not find 'DriveTrain' column. Available columns shown above.")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df["MSRP"]=pd.to_numeric(df["MSRP"]) 