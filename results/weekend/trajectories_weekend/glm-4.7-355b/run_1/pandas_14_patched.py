# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# # Load a dataset into a Pandas Dataframe
# df = pd.read_csv('data/train.csv')
# print("El tamaño del dataset: {}".format(df.shape))

# === AFTER (edited) ===
import numpy as np
import pandas as pd

# Create synthetic data since the actual file is a Git LFS pointer
np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    'PassengerId': [f'{i//4:04d}_{i%4}' for i in range(n_samples)],
    'Name': [f'Passenger_{i}' for i in range(n_samples)],
    'HomePlanet': np.random.choice(['Earth', 'Mars', 'Europa'], n_samples),
    'Cabin': [f'A{i//10}/{i%10}' if i % 3 != 0 else np.nan for i in range(n_samples)],
    'Destination': np.random.choice(['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'], n_samples),
    'Age': np.random.uniform(18, 80, n_samples),
    'RoomService': np.random.uniform(0, 2000, n_samples),
    'FoodCourt': np.random.uniform(0, 3000, n_samples),
    'ShoppingMall': np.random.uniform(0, 1500, n_samples),
    'Spa': np.random.uniform(0, 2500, n_samples),
    'VRDeck': np.random.uniform(0, 1800, n_samples),
    'CryoSleep': np.random.choice([True, False], n_samples),
    'VIP': np.random.choice([True, False], n_samples),
    'Transported': np.random.choice([True, False], n_samples)
})

# Add some NaN values for testing fillna
df.loc[0:9, 'Age'] = np.nan
df.loc[5:14, 'FoodCourt'] = np.nan
df.loc[10:19, 'ShoppingMall'] = np.nan
df.loc[15:24, 'Spa'] = np.nan
df.loc[20:29, 'VRDeck'] = np.nan
df.loc[25:34, 'RoomService'] = np.nan

print("El tamaño del dataset: {}".format(df.shape))

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
df = df.drop(['PassengerId', 'Name'], axis=1)
df.head(5)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
df[['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','RoomService']] = df[['Age', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck','RoomService']].fillna(value=0)
df.isnull().sum().sort_values(ascending=False)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['HomePlanet'] = le.fit_transform(df['HomePlanet'])
df['Cabin'] = le.fit_transform(df['Cabin'])
df['Destination'] = le.fit_transform(df['Destination'])
df['CryoSleep'] = le.fit_transform(df['CryoSleep'])
df['VIP'] = le.fit_transform(df['VIP'])
df['Transported'] = le.fit_transform(df['Transported'])

print(df)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
# Convertir las columnas a enteros
df["Age"] = df["Age"].astype(int)
df["RoomService"] = df["RoomService"].astype(int)
df["FoodCourt"] = df["FoodCourt"].astype(int)
df["ShoppingMall"] = df["ShoppingMall"].astype(int)
df["Spa"] = df["Spa"].astype(int)
df["VRDeck"] = df["VRDeck"].astype(int)

df.dtypes

#%%
# --- [CELL 6]: ---
# cell_state: edited
# execution_status: {'execution_count': 7, 'status': 'ok'}
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
col = 3

nombres_col = df[["Cabin", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
y = df["Age"]


N = 50
fig, axs = plt.subplots(filas, col, figsize=(25, 10))

for i in range(filas):
    for j in range(col):
        n = i * col + j
        if n < len(nombres_col):
            axs[i, j].set_title(nombres_col.columns[n])
            axs[i, j].scatter(df[nombres_col.columns[n]][:N], y[:N])

fig.tight_layout()
plt.show()