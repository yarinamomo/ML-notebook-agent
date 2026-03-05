# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import numpy as np # linear algebra
# import pandas as pd

# === AFTER (edited) ===
import numpy as np
import pandas as pd

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df=pd.read_csv("data/2. Cars Data1.csv")

# === AFTER (edited) ===
df = pd.read_csv("data/2. Cars Data1.csv")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df["DriveTrain"]=df["DriveTrain"].astype("category")

# === AFTER (edited) ===
df["DriveTrain"] = df["DriveTrain"].astype("category")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df["MSRP"]=pd.to_numeric(df["MSRP"]) 

# === AFTER (edited) ===
df["MSRP"] = pd.to_numeric(df["MSRP"])