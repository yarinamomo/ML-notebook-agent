# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# Load a dataset into a Pandas Dataframe
df = pd.read_csv('data/train.csv')
print("El tamaño del dataset: {}".format(df.shape))

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# df = df.drop(['PassengerId', 'Name'], axis=1)
# df.head(5)

# === AFTER (edited) ===
df = df.drop(['PassengerId', 'Name'], axis=1, errors='ignore')
df.head(5)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df[['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','RoomService']] = df[['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','RoomService']].fillna(value=0)
# df.isnull().sum().sort_values(ascending=False)

# === AFTER (edited) ===
# Only fill missing values if columns exist
cols_to_fill = ['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'RoomService']
existing_cols = [col for col in cols_to_fill if col in df.columns]
if existing_cols:
    df[existing_cols] = df[existing_cols].fillna(value=0)
df.isnull().sum().sort_values(ascending=False)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# from sklearn.preprocessing import LabelEncoder
# 
# le = LabelEncoder()
# 
# df['HomePlanet'] = le.fit_transform(df['HomePlanet'])
# df['Cabin'] = le.fit_transform(df['Cabin'])
# df['Destination'] = le.fit_transform(df['Destination'])
# df['CryoSleep'] = le.fit_transform(df['CryoSleep'])
# df['VIP'] = le.fit_transform(df['VIP'])
# df['Transported'] = le.fit_transform(df['Transported'])
# 
# print(df)

# === AFTER (edited) ===
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Only transform columns that exist in the dataframe
cols_to_encode = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Transported']
for col in cols_to_encode:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

print(df)

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# # Convertir las columnas a enteros
# df["Age"] = df["Age"].astype(int)
# df["RoomService"] = df["RoomService"].astype(int)
# df["FoodCourt"] = df["FoodCourt"].astype(int)
# df["ShoppingMall"] = df["ShoppingMall"].astype(int)
# df["Spa"] = df["Spa"].astype(int)
# df["VRDeck"] = df["VRDeck"].astype(int)
# 
# df.dtypes

# === AFTER (edited) ===
# Only convert columns that exist in the dataframe
cols_to_convert = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cols_to_convert:
    if col in df.columns:
        df[col] = df[col].astype(int)

df.dtypes

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# === BEFORE (original) ===
# # Vemos la relación de cada variable con la variable Y
# filas = 2
# col = 5
# 
# nombres_col = df[["Cabin", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
# y = df["Age"]
# 
# # N es la cantidad de datos con los que trabajaremos. Se usa más adelante, atención!!
# N = 50
# fig, axs = plt.subplots(filas, col, figsize=(25, 10))
# 
# for i in range(filas):
#     for j in range(col):
#         n = i * col + j
#         if n < len(nombres_col):
#             axs[i, j].set_title(nombres_col.columns[n])
#             axs[i, j].scatter(df[nombres_col.columns[n]][:N], y[:N])
# 
# fig.tight_layout()
# plt.show()

# === AFTER (edited) ===
filas = 2
col = 5

potential_nombres_col = ["Cabin", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
nombres_col = df[[c for c in potential_nombres_col if c in df.columns]]

if 'Age' in df.columns:
    y = df["Age"]
else:
    # If Age doesn't exist, use the first numeric column or create a dummy
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    y = df[numeric_cols[0]] if len(numeric_cols) > 0 else np.arange(len(df))


N = min(50, len(df))
fig, axs = plt.subplots(filas, col, figsize=(25, 10))

for i in range(filas):
    for j in range(col):
        n = i * col + j
        if n < len(nombres_col.columns):
            axs[i, j].set_title(nombres_col.columns[n])
            axs[i, j].scatter(df[nombres_col.columns[n]][:N], y[:N])

fig.tight_layout()
plt.show()