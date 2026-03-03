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

# Columns to encode
cols_to_encode = ['HomePlanet', 'Cabin', 'Destination', 'CryoSleep', 'VIP', 'Transported']

for col in cols_to_encode:
    try:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
    except Exception as e:
        print(f"Warning: Could not encode column '{col}': {e}")

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
# Convert columns to int if they exist
cols_to_convert = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in cols_to_convert:
    try:
        if col in df.columns:
            df[col] = df[col].astype(int)
    except Exception as e:
        print(f"Warning: Could not convert column '{col}' to int: {e}")

df.dtypes

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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

# Only use columns that exist in the dataframe
cols_to_plot = ["Cabin", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
existing_cols = [col for col in cols_to_plot if col in df.columns]

if len(existing_cols) > 0:
    nombres_col = df[existing_cols]
else:
    nombres_col = df[[]]  # Empty dataframe

# Check if Age exists, otherwise skip
if "Age" in df.columns:
    y = df["Age"]
else:
    y = None

N = 50
fig, axs = plt.subplots(filas, col, figsize=(25, 10))

for i in range(filas):
    for j in range(col):
        n = i * col + j
        if n < len(nombres_col.columns):
            axs[i, j].set_title(nombres_col.columns[n])
            if y is not None:
                axs[i, j].scatter(df[nombres_col.columns[n]][:N], y[:N])
        else:
            ax.set_visible(False) if len(nombres_col.columns) <= i*col+j+1 else None
            axs[i, j].axis('off')

fig.tight_layout()
plt.show()