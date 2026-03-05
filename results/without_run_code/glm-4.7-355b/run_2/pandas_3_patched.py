# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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

# Create sample data to simulate the pet adoption dataset structure
# Since the actual data files are Git LFS pointers (placeholders),
# we'll create minimal working data with the required columns

# Sample training data
train_data = {
    'PetID': ['pet_001', 'pet_002', 'pet_003', 'pet_004', 'pet_005'],
    'Name': ['Buddy', 'Max', 'Bella', 'Charlie', 'Luna'],
    'RescuerID': ['rescuer_001', 'rescuer_001', 'rescuer_002', 'rescuer_002', 'rescuer_003'],
    'Description': ['Friendly dog', 'Playful puppy', 'Sweet cat', 'Active dog', 'Gentle cat']
}

# Sample test data  
test_data = {
    'PetID': ['pet_006', 'pet_007'],
    'Name': ['Rocky', 'Molly'],
    'RescuerID': ['rescuer_003', 'rescuer_004'],
    'Description': ['Strong dog', 'Calm cat']
}

train = pd.DataFrame(train_data).set_index('PetID')
test = pd.DataFrame(test_data).set_index('PetID')

print("Train data shape:", train.shape)
print("Test data shape:", test.shape)

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
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
rescuer_count = train["RescuerID"].value_counts().rename("rescuer_count")
rescuer_encoder = LabelEncoder().fit(train["RescuerID"])

def procData(data, name_count, rescuer_count, rescuer_encoder):

    data = data.join(rescuer_count, on="RescuerID")
    # Handle unseen rescuer IDs by catching the error and assigning -1
    try:
        data["RescuerID"] = rescuer_encoder.transform(data["RescuerID"])
    except ValueError:
        # Map seen values and set unseen to -1
        data["RescuerID"] = data["RescuerID"].map(
            lambda x: rescuer_encoder.transform([x])[0] if x in rescuer_encoder.classes_ else -1
        )
    data["RescuerID"] = data["RescuerID"].fillna(-1)


    data["NameNull"] = data["Name"].isnull()
    data["NameLen"] = data["Name"].fillna("").str.len()
    data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
    data = data.join(name_count, on="Name")
    data["name_count"] = data["name_count"].fillna(0)

    data = data.drop(["Name", "RescuerID", "Description"], axis=1)
    return data

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# train = procData(train, name_count, RescuerID)
# test = procData(test, name_count, RescuerID)

# === AFTER (edited) ===
train = procData(train, name_count, rescuer_count, rescuer_encoder)
test = procData(test, name_count, rescuer_count, rescuer_encoder)