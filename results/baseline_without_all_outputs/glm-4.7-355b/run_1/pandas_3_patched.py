# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
name_count = train["Name"].value_counts().rename("name_count")
RescuerID = LabelEncoder().fit(train["RescuerID"])

def procData(data, name_count, RescuerID):
    # Ambas dos opciones de abajo estan bien para reformular los RescuerID
    data = data.join(name_count, on ="RescuerID")
    data["RescuerID"] = RescuerID.transform(data["RescuerID"])
    data["RescuerID"] = data["RescuerID"].fillna(-1)

    #Para los nombres
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
# Fix procData to handle unseen values and correct join logic
def procData(data, name_count, RescuerID):
    # Handle unseen RescuerID values by mapping to -1
    rescuer_id_map = {v: i for i, v in enumerate(RescuerID.classes_)}
    data["RescuerID"] = data["RescuerID"].map(lambda x: rescuer_id_map.get(x, -1))
    
    data["NameNull"] = data["Name"].isnull()
    data["NameLen"] = data["Name"].fillna("").str.len()
    data["SinNombre"] = data["Name"].str.lower().replace(" ", "") == "nonameyet"
    
    # Map name_count on Name, using 0 for unseen names
    data["name_count"] = data["Name"].map(name_count).fillna(0)

    data = data.drop(["Name", "RescuerID", "Description"], axis=1)
    return data

train = procData(train, name_count, RescuerID)
test = procData(test, name_count, RescuerID)