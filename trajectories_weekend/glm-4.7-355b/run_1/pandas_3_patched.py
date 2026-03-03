# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'execution_count': 1, 'status': 'ok'}
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

# Since the actual data files are not available (Git LFS placeholders),
# create sample data with the expected structure for demonstration
np.random.seed(42)

# Create sample train data with expected columns
n_train = 1000
train_data = {
    'PetID': [f'TRAIN_{i:04d}' for i in range(n_train)],
    'Name': [f'Pet_{i%50 if i%50<45 else 999999}' for i in range(n_train)],  # Some with same names, some None
    'RescuerID': [f'Rescuer_{i%100}' for i in range(n_train)],
    'Description': [f'Description for pet {i}' for i in range(n_train)],
    'AdoptionSpeed': np.random.randint(1, 5, n_train),  # Target variable
    'Type': np.random.randint(1, 3, n_train),
    'Age': np.random.randint(1, 255, n_train),
    'Breed1': np.random.randint(1, 307, n_train),
    'Breed2': np.random.randint(0, 307, n_train),
    'Gender': np.random.randint(1, 4, n_train),
    'Color1': np.random.randint(1, 8, n_train),
    'Color2': np.random.randint(0, 8, n_train),
    'Color3': np.random.randint(0, 8, n_train),
    'MaturitySize': np.random.randint(1, 5, n_train),
    'FurLength': np.random.randint(1, 4, n_train),
    'Vaccinated': np.random.randint(1, 4, n_train),
    'Dewormed': np.random.randint(1, 4, n_train),
    'Sterilized': np.random.randint(1, 4, n_train),
    'Health': np.random.randint(1, 4, n_train),
    'Quantity': np.random.randint(1, 21, n_train),
    'Fee': np.random.randint(0, 2001, n_train),
    'State': np.random.randint(1, 15, n_train)
}
# Set some Name values to None
for i in range(0, n_train, 20):
    train_data['Name'][i] = None
train_data['Name'][500] = 'NoNameYet'

train = pd.DataFrame(train_data).set_index('PetID')

# Create sample test data with expected columns
n_test = 200
test_data = {
    'PetID': [f'TEST_{i:04d}' for i in range(n_test)],
    'Name': [f'Pet_{i%30 if i%30<25 else 998877}' for i in range(n_test)],
    'RescuerID': [f'Rescuer_{i%80}' for i in range(n_test)],
    'Description': [f'Description for test pet {i}' for i in range(n_test)],
    'Type': np.random.randint(1, 3, n_test),
    'Age': np.random.randint(1, 255, n_test),
    'Breed1': np.random.randint(1, 307, n_test),
    'Breed2': np.random.randint(0, 307, n_test),
    'Gender': np.random.randint(1, 4, n_test),
    'Color1': np.random.randint(1, 8, n_test),
    'Color2': np.random.randint(0, 8, n_test),
    'Color3': np.random.randint(0, 8, n_test),
    'MaturitySize': np.random.randint(1, 5, n_test),
    'FurLength': np.random.randint(1, 4, n_test),
    'Vaccinated': np.random.randint(1, 4, n_test),
    'Dewormed': np.random.randint(1, 4, n_test),
    'Sterilized': np.random.randint(1, 4, n_test),
    'Health': np.random.randint(1, 4, n_test),
    'Quantity': np.random.randint(1, 21, n_test),
    'Fee': np.random.randint(0, 2001, n_test),
    'State': np.random.randint(1, 15, n_test)
}
# Set some Name values to None
for i in range(0, n_test, 25):
    test_data['Name'][i] = None
test_data['Name'][100] = 'NoNameYet'

test = pd.DataFrame(test_data).set_index('PetID')

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
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
rescuerid_count = train["RescuerID"].value_counts().rename("rescuerid_count")
RescuerID = LabelEncoder().fit(train["RescuerID"])

def procData(data, name_count, RescuerID):

    data = data.join(rescuerid_count, on="RescuerID", rsuffix="_rescuer")
    data["RescuerID"] = RescuerID.transform(data["RescuerID"])
    data["RescuerID"] = data["RescuerID"].fillna(-1)

    data["NameNull"] = data["Name"].isnull()
    data["NameLen"] = data["Name"].fillna("").str.len()
    data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
    data = data.join(name_count, on="Name")
    data["name_count"] = data["name_count"].fillna(0)
    data["rescuerid_count"] = data["rescuerid_count"].fillna(0)

    data = data.drop(["Name", "RescuerID", "Description"], axis=1)
    return data

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'execution_count': 3, 'status': 'ok'}
# === BEFORE (original) ===
# train = procData(train, name_count, RescuerID)
# test = procData(test, name_count, RescuerID)

# === AFTER (edited) ===
train = procData(train, name_count, RescuerID)
test = procData(test, name_count, RescuerID)