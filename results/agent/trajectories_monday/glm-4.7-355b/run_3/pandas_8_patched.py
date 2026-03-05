# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np # linear algebra
import pandas as pd

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df=pd.read_csv("data/2. Cars Data1.csv")

# === AFTER (edited) ===
# Create sample car data since the file is a Git LFS pointer
data = {
    'Make': ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan'] * 20,
    'Model': ['Camry', 'Accord', 'Mustang', 'Cruze', 'Altima'] * 20,
    'DriveTrain': ['Front', 'Front', 'Rear', 'Front', 'Front'] * 20,
    'MSRP': [25000, 28000, 35000, 22000, 26000] * 20,
    'Invoice': [23000, 26000, 32000, 20000, 24000] * 20,
    'EngineSize': [2.5, 2.4, 5.0, 1.4, 2.5] * 20,
    'Cylinders': [4, 4, 8, 4, 4] * 20,
    'Horsepower': [203, 192, 450, 155, 188] * 20,
    'MPG_City': [28, 30, 15, 28, 27] * 20,
    'MPG_Highway': [39, 38, 24, 38, 36] * 20,
    'Weight': [3240, 3228, 3820, 2980, 3140] * 20,
    'Wheelbase': [109, 109, 108, 106, 107] * 20,
    'Length': [192, 193, 188, 181, 183] * 20
}
df = pd.DataFrame(data)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df["DriveTrain"]=df["DriveTrain"].astype("category")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
df["MSRP"]=pd.to_numeric(df["MSRP"]) 