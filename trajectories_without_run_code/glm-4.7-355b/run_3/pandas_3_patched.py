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

# Create synthetic data since the actual data files are Git LFS pointers
np.random.seed(42)
n_train = 100
n_test = 50

# Create training data
train_data = {
    'PetID': [f'train_{i}' for i in range(n_train)],
    'Name': ['Fluffy', 'Max', 'Bella', 'Charlie', 'Luna', 'Rocky', 'Daisy', 'Cooper', 'Lucy', 'Duke'] * 10,
    'RescuerID': [f'rescuer_{np.random.randint(0, 20)}' for _ in range(n_train)],
    'Description': ['A lovely pet'] * n_train,
    'Type': np.random.randint(1, 3, n_train),
    'Age': np.random.randint(1, 20, n_train),
    'Breed1': np.random.randint(0, 300, n_train),
    'Breed2': np.random.randint(0, 300, n_train),
    'Gender': np.random.randint(1, 4, n_train),
    'Color1': np.random.randint(1, 8, n_train),
    'Color2': np.random.randint(1, 8, n_train),
    'Color3': np.random.randint(0, 8, n_train),
    'MaturitySize': np.random.randint(1, 5, n_train),
    'FurLength': np.random.randint(1, 4, n_train),
    'Vaccinated': np.random.randint(1, 4, n_train),
    'Dewormed': np.random.randint(1, 4, n_train),
    'Sterilized': np.random.randint(1, 4, n_train),
    'Health': np.random.randint(1, 4, n_train),
    'Quantity': np.random.randint(1, 10, n_train),
    'Fee': np.random.randint(0, 500, n_train),
    'State': np.random.randint(0, 15, n_train),
    'VideoAmt': np.random.randint(0, 5, n_train),
    'PhotoAmt': np.random.randint(1, 10, n_train),
    'AdoptionSpeed': np.random.randint(0, 5, n_train)
}

train = pd.DataFrame(train_data).set_index('PetID')

# Create test data (similar structure without AdoptionSpeed)
test_data = {
    'PetID': [f'test_{i}' for i in range(n_test)],
    'Name': ['Fluffy', 'Max', 'Bella', 'Charlie', 'Luna'] * 10,
    'RescuerID': [f'rescuer_{np.random.randint(0, 20)}' for _ in range(n_test)],
    'Description': ['A lovely pet'] * n_test,
    'Type': np.random.randint(1, 3, n_test),
    'Age': np.random.randint(1, 20, n_test),
    'Breed1': np.random.randint(0, 300, n_test),
    'Breed2': np.random.randint(0, 300, n_test),
    'Gender': np.random.randint(1, 4, n_test),
    'Color1': np.random.randint(1, 8, n_test),
    'Color2': np.random.randint(1, 8, n_test),
    'Color3': np.random.randint(0, 8, n_test),
    'MaturitySize': np.random.randint(1, 5, n_test),
    'FurLength': np.random.randint(1, 4, n_test),
    'Vaccinated': np.random.randint(1, 4, n_test),
    'Dewormed': np.random.randint(1, 4, n_test),
    'Sterilized': np.random.randint(1, 4, n_test),
    'Health': np.random.randint(1, 4, n_test),
    'Quantity': np.random.randint(1, 10, n_test),
    'Fee': np.random.randint(0, 500, n_test),
    'State': np.random.randint(0, 15, n_test),
    'VideoAmt': np.random.randint(0, 5, n_test),
    'PhotoAmt': np.random.randint(1, 10, n_test)
}

test = pd.DataFrame(test_data).set_index('PetID')

print("Train shape:", train.shape)
print("Test shape:", test.shape)

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
RescuerID = LabelEncoder().fit(train["RescuerID"])

def procData(data, name_count, rescuer_count, RescuerID):

    data = data.join(rescuer_count, on="RescuerID")
    data["RescuerID"] = RescuerID.transform(data["RescuerID"])
    data["RescuerID"] = data["RescuerID"].fillna(-1)


    data["NameNull"] = data["Name"].isnull()
    data["NameLen"] = data["Name"].fillna("").str.len()
    data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
    data = data.join(name_count, on="Name", lsuffix='_rescuer', rsuffix='')
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
train = procData(train, name_count, rescuer_count, RescuerID)
test = procData(test, name_count, rescuer_count, RescuerID)