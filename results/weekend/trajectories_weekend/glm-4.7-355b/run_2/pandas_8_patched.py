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
import numpy as np
import pandas as pd

# Create a mock cars dataset since the original file is a Git LFS pointer
data = {
    'DriveTrain': ['Front', 'Rear', 'Front', 'Front', 'Rear', 'All', 'Front', 'Rear', 'All', 'Front',
                   'Rear', 'Front', 'All', 'Front', 'Rear', 'Front', 'All', 'Rear', 'Front', 'All'],
    'MSRP': [21500, 28900, 22400, 21750, 31200, 34500, 22800, 29500, 36000, 21900,
             28500, 22000, 35000, 21600, 30000, 22300, 35500, 29000, 21800, 37000],
    'Make': ['Toyota', 'BMW', 'Honda', 'Toyota', 'BMW', 'Audi', 'Honda', 'BMW', 'Audi', 'Toyota',
             'BMW', 'Honda', 'Audi', 'Toyota', 'BMW', 'Honda', 'Audi', 'BMW', 'Toyota', 'Audi'],
    'Model': ['Camry', '3 Series', 'Accord', 'Corolla', '5 Series', 'A4', 'Civic', 'X3', 'Q5', 'Prius',
              'X5', 'CR-V', 'Q7', 'RAV4', '7 Series', 'Fit', 'R8', 'Z4', 'Yaris', 'TT']
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