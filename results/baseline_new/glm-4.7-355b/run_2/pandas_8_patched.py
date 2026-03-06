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
# Try to find DriveTrain column (handle potential case/spacing variations)
if 'DriveTrain' in df.columns:
    df["DriveTrain"]=df["DriveTrain"].astype("category")
elif 'Drivetrain' in df.columns:
    df["Drivetrain"]=df["Drivetrain"].astype("category")
elif 'drivetrain' in df.columns:
    df["drivetrain"]=df["drivetrain"].astype("category")
elif 'Drive Train' in df.columns:
    df["Drive Train"]=df["Drive Train"].astype("category")
elif 'drive train' in df.columns:
    df["drive train"]=df["drive train"].astype("category")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df["MSRP"]=pd.to_numeric(df["MSRP"]) 

# === AFTER (edited) ===
# Try to find MSRP column (handle potential case/spacing variations)
if 'MSRP' in df.columns:
    df["MSRP"]=pd.to_numeric(df["MSRP"])
elif 'msrp' in df.columns:
    df["msrp"]=pd.to_numeric(df["msrp"])
elif 'Msrp' in df.columns:
    df["Msrp"]=pd.to_numeric(df["Msrp"])
elif 'M.S.R.P.' in df.columns:
    df["M.S.R.P."]=pd.to_numeric(df["M.S.R.P."])