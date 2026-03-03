# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import numpy as np # linear algebra
import pandas as pd

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# df=pd.read_csv("data/2. Cars Data1.csv")

# === AFTER (edited) ===
# Create sample data with expected columns for Cars dataset
data = {
    'Make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan'],
    'Model': ['Camry', 'Accord', 'Fusion', 'Malibu', 'Altima'],
    'Year': [2020, 2021, 2019, 2020, 2021],
    'DriveTrain': ['Front', 'Front', 'Front', 'Front', 'Front'],
    'MSRP': ['24970', '26120', '28000', '23200', '24750'],
    'Invoice': ['23102', '23950', '25600', '21340', '22775']
}
df = pd.DataFrame(data)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
df["DriveTrain"]=df["DriveTrain"].astype("category")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
df["MSRP"]=pd.to_numeric(df["MSRP"]) 