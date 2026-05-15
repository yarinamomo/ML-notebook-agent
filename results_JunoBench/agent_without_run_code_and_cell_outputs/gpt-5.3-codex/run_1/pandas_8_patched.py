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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df["DriveTrain"]=df["DriveTrain"].astype("category")

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df["MSRP"]=pd.to_numeric(df["MSRP"])

# === AFTER (edited) ===
df["MSRP"]=pd.to_numeric(df["MSRP"].replace(r"[\$,]", "", regex=True).str.strip())