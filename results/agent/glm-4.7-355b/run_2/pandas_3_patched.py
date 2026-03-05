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

# Create sample synthetic data since actual files are not available
# (files are Git LFS pointers, not the actual data)
np.random.seed(42)
n_train = 1000
n_test = 400

# Generate sample data with the expected columns
train_data = {
    'PetID': [f'dog_{i}' for i in range(n_train)],
    'Name': np.random.choice(['Buddy', 'Max', 'Charlie', 'Luna', 'Rocky', 'Coco', 'NoNameYet', None], n_train),
    'RescuerID': [f'rescuer_{np.random.randint(1, 51)}' for _ in range(n_train)],
    'Description': [f'A cute pet description number {i}' for i in range(n_train)],
    'Type': np.random.choice([1, 2], n_train),  # 1=dog, 2=cat
    'Age': np.random.randint(1, 60, n_train),
    'Breed1': np.random.randint(1, 100, n_train),
    'Breed2': np.random.randint(0, 100, n_train),
    'Gender': np.random.choice([1, 2, 3], n_train),
    'Color1': np.random.randint(1, 10, n_train),
    'Color2': np.random.randint(0, 10, n_train),
    'Color3': np.random.randint(0, 10, n_train),
    'MaturitySize': np.random.choice([1, 2, 3, 4], n_train),
    'FurLength': np.random.choice([1, 2, 3], n_train),
    'Vaccinated': np.random.choice([1, 2, 3], n_train),
    'Dewormed': np.random.choice([1, 2, 3], n_train),
    'Sterilized': np.random.choice([1, 2, 3], n_train),
    'Health': np.random.choice([1, 2, 3], n_train),
    'Quantity': np.random.randint(1, 5, n_train),
    'Fee': np.random.randint(0, 500, n_train),
    'State': np.random.randint(1, 17, n_train),
    'AdoptionSpeed': np.random.randint(0, 4, n_train),
}

test_data = {
    'PetID': [f'dog_{i}' for i in range(n_train, n_train + n_test)],
    'Name': np.random.choice(['Buddy', 'Max', 'Charlie', 'Luna', 'Rocky', 'Coco', 'NoNameYet', None], n_test),
    'RescuerID': [f'rescuer_{np.random.randint(1, 51)}' for _ in range(n_test)],
    'Description': [f'A cute pet description number {i}' for i in range(n_test)],
    'Type': np.random.choice([1, 2], n_test),
    'Age': np.random.randint(1, 60, n_test),
    'Breed1': np.random.randint(1, 100, n_test),
    'Breed2': np.random.randint(0, 100, n_test),
    'Gender': np.random.choice([1, 2, 3], n_test),
    'Color1': np.random.randint(1, 10, n_test),
    'Color2': np.random.randint(0, 10, n_test),
    'Color3': np.random.randint(0, 10, n_test),
    'MaturitySize': np.random.choice([1, 2, 3, 4], n_test),
    'FurLength': np.random.choice([1, 2, 3], n_test),
    'Vaccinated': np.random.choice([1, 2, 3], n_test),
    'Dewormed': np.random.choice([1, 2, 3], n_test),
    'Sterilized': np.random.choice([1, 2, 3], n_test),
    'Health': np.random.choice([1, 2, 3], n_test),
    'Quantity': np.random.randint(1, 5, n_test),
    'Fee': np.random.randint(0, 500, n_test),
    'State': np.random.randint(1, 17, n_test),
}

train = pd.DataFrame(train_data).set_index('PetID')
test = pd.DataFrame(test_data).set_index('PetID')

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
RescuerID = LabelEncoder().fit(train["RescuerID"])

def procData(data, name_count, RescuerID):

    data = data.join(name_count, on ="RescuerID")
    data["RescuerID"] = RescuerID.transform(data["RescuerID"])
    data["RescuerID"] = data["RescuerID"].fillna(-1)


    data["NameNull"] = data["Name"].isnull()
    data["NameLen"] = data["Name"].fillna("").str.len()
    data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
    data = data.join(name_count, on="Name", rsuffix="_name")
    data["name_count_name"] = data["name_count_name"].fillna(0)

    data = data.drop(["Name", "RescuerID", "Description"], axis=1)
    return data

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
train = procData(train, name_count, RescuerID)
test = procData(test, name_count, RescuerID)