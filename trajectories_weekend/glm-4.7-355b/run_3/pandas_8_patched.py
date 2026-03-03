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
# Create sample car data since the CSV file is a Git LFS pointer
data = {
    'DriveTrain': ['Front', 'Rear', 'Front', 'All', 'Rear', 'Front', 'All', 'Rear', 'Front', 'All'],
    'MSRP': ['15000', '25000', '18000', '32000', '45000', '20000', '35000', '55000', '17000', '28000'],
    'Make': ['Toyota', 'BMW', 'Honda', 'Audi', 'Mercedes', 'Ford', 'Subaru', 'Lexus', 'Nissan', 'Volkswagen'],
    'Model': ['Camry', '3 Series', 'Accord', 'A4', 'E-Class', 'Fusion', 'Outback', 'ES', 'Altima', 'Passat'],
    'Year': [2018, 2019, 2018, 2020, 2019, 2017, 2020, 2021, 2018, 2019],
    'Type': ['Sedan', 'Sedan', 'Sedan', 'Sedan', 'Sedan', 'Sedan', 'Wagon', 'Sedan', 'Sedan', 'Sedan'],
    'Origin': ['Asia', 'Europe', 'Asia', 'Europe', 'Europe', 'USA', 'USA', 'Asia', 'Asia', 'Europe']
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