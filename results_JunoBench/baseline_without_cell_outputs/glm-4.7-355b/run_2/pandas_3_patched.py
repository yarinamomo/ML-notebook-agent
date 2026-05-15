# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics

train = pd.read_csv('data/train.csv.zip', index_col="PetID")
test = pd.read_csv('data/test.csv.zip', index_col="PetID")

# Ambas dos opciones de abajo estan bien para reformular los RescuerID
#data = data.join(data["RescuerID"].value_counts().rename("rescuent_count"), on ="ResucerID")
# data["RescuerID"] = data["RescuerID"].astype("category").cat.codes

#Para los nombres
# data["NameNull"] = data["Name"].isnull()
# data["NameLen"] = data["Name"].fillna("").str.len()
# data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
# data = data.join(data["Name"].value_counts().rename("name_count"), on="Name")
# data["name_count"] = data["name_count"].fillna(0)

# data = data.drop(["Name", "RescuerID", "Description"], axis=1)


# data

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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
rescuer_name_count = train["Name"].value_counts().rename("rescuer_name_count")
pet_name_count = train["Name"].value_counts().rename("pet_name_count")
RescuerID = LabelEncoder().fit(train["RescuerID"])

def procData(data, rescuer_name_count, pet_name_count, RescuerID):

    data = data.join(rescuer_name_count, on ="RescuerID")
    data["RescuerID"] = RescuerID.transform(data["RescuerID"])
    data["RescuerID"] = data["RescuerID"].fillna(-1)


    data["NameNull"] = data["Name"].isnull()
    data["NameLen"] = data["Name"].fillna("").str.len()
    data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
    data = data.join(pet_name_count, on="Name")
    data["pet_name_count"] = data["pet_name_count"].fillna(0)
    data["rescuer_name_count"] = data["rescuer_name_count"].fillna(0)

    data = data.drop(["Name", "RescuerID", "Description"], axis=1)
    return data

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train = procData(train, name_count, RescuerID)
test = procData(test, name_count, RescuerID)