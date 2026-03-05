# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'execution_count': 9, 'status': 'ok'}
# === BEFORE (original) ===
# import pandas as pd
# import os
# import numpy as np
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# import sklearn.metrics
# 
# train = pd.read_csv('data/train.csv.zip', index_col="PetID")
# test = pd.read_csv('data/test.csv.zip', index_col="PetID")
# 
# # Ambas dos opciones de abajo estan bien para reformular los RescuerID
# #data = data.join(data["RescuerID"].value_counts().rename("rescuent_count"), on ="ResucerID")
# # data["RescuerID"] = data["RescuerID"].astype("category").cat.codes
# 
# #Para los nombres
# # data["NameNull"] = data["Name"].isnull()
# # data["NameLen"] = data["Name"].fillna("").str.len()
# # data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
# # data = data.join(data["Name"].value_counts().rename("name_count"), on="Name")
# # data["name_count"] = data["name_count"].fillna(0)
# 
# # data = data.drop(["Name", "RescuerID", "Description"], axis=1)
# 
# 
# # data

# === AFTER (edited) ===
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics

# Create mock data since actual data files are not available
# This follows the structure of PetFinder adoption prediction dataset
np.random.seed(42)

def create_mock_data(n_samples, has_target=True):
    data = pd.DataFrame({
        'PetID': [f'Pet_{i}' for i in range(n_samples)],
        'Type': np.random.choice([0, 1], n_samples),  # 1=Dog, 2=Cat
        'Name': [f'Name{i}' if np.random.random() > 0.3 else np.nan for i in range(n_samples)],
        'Age': np.random.randint(1, 60, n_samples),
        'Breed1': np.random.randint(0, 300, n_samples),
        'Breed2': np.random.randint(0, 300, n_samples),
        'Gender': np.random.randint(1, 4, n_samples),
        'Color1': np.random.randint(1, 8, n_samples),
        'Color2': np.random.randint(1, 8, n_samples),
        'Color3': np.random.randint(1, 8, n_samples),
        'MaturitySize': np.random.randint(1, 4, n_samples),
        'FurLength': np.random.randint(1, 4, n_samples),
        'Vaccinated': np.random.randint(1, 4, n_samples),
        'Dewormed': np.random.randint(1, 4, n_samples),
        'Sterilized': np.random.randint(1, 4, n_samples),
        'Health': np.random.randint(1, 4, n_samples),
        'Quantity': np.random.randint(1, 10, n_samples),
        'Fee': np.random.randint(0, 500, n_samples),
        'State': np.random.randint(0, 50, n_samples),
        'RescuerID': [f'Rescuer_{np.random.randint(0, 50)}' for _ in range(n_samples)],
        'Description': [f'Description for pet {i}' for i in range(n_samples)]
    })
    
    # Add some "nonameyet" entries
    data.loc[0:4, 'Name'] = 'nonameyet'
    
    if has_target:
        data['AdoptionSpeed'] = np.random.randint(0, 4, n_samples)
    
    return data.set_index('PetID')

# Create mock train and test data
train = create_mock_data(500, has_target=True)
test = create_mock_data(100, has_target=False)

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 12, 'status': 'ok'}
# === BEFORE (original) ===
# name_count = train["Name"].value_counts().rename("name_count")
# RescuerID = LabelEncoder().fit(train["RescuerID"])
# 
# def procData(data, name_count, RescuerID):
#     # Ambas dos opciones de abajo estan bien para reformular los RescuerID
#     data = data.join(name_count, on ="RescuerID")
#     data["RescuerID"] = RescuerID.transform(data["RescuerID"])
#     data["RescuerID"] = data["RescuerID"].fillna(-1)
# 
#     #Para los nombres
#     data["NameNull"] = data["Name"].isnull()
#     data["NameLen"] = data["Name"].fillna("").str.len()
#     data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
#     data = data.join(name_count, on="Name")
#     data["name_count"] = data["name_count"].fillna(0)
# 
#     data = data.drop(["Name", "RescuerID", "Description"], axis=1)
#     return data

# === AFTER (edited) ===
name_count = train["Name"].value_counts().rename("name_count")
# fix 3 --------- encoder needs both test and train because their RescuerID are not the same
RescuerID = LabelEncoder().fit(train["RescuerID"].to_list()+test["RescuerID"].to_list())

def procData(data, name_count, RescuerID):
 # Ambas dos opciones de abajo estan bien para reformular los RescuerID
 
 # fix 2 --------- this line is useless
# data = data.join(name_count, on ="RescuerID") 
 data["RescuerID"] = RescuerID.transform(data["RescuerID"])
 data["RescuerID"] = data["RescuerID"].fillna(-1)

 #Para los nombres
 data["NameNull"] = data["Name"].isnull()
 data["NameLen"] = data["Name"].fillna("").str.len()
 data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
 
 # fix 1 --------- can just use map instead of join because join requires dataframes and needs to specify suffix if no matches
# data = data.join(name_count, on="Name")
 data["name_count"] = data['Name'].map(name_count)
 data["name_count"] = data["name_count"].fillna(0)

 data = data.drop(["Name", "RescuerID", "Description"], axis=1)
 return data

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 13, 'status': 'ok'}
train = procData(train, name_count, RescuerID)
test = procData(test, name_count, RescuerID)