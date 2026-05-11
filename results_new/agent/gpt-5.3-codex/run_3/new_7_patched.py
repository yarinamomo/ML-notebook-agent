# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
filename = "data/allStar-dr17-synspec_rev1.fits"
data = fits.open(filename)[1].data
print("Loaded Fits")
features = []
# for i in data.columns:
#     if data[i.name].shape == (733901,):
#         features.append(i.name)


print("Loaded Data")


IDS = []

toUse = []
flags = [
    'APOGEE_ID', 'APOGEE2_TARGET2'
]
features = toUse + flags


df = pd.DataFrame()
features=[] + IDS + features

features = list(set(features))
for feature in features:
    print(f"Adding {feature}...", end=" ")
    df[feature] = data[feature]
    print("Done")

# Handle endia\nness issues
for col in df.columns:
    if hasattr(df[col], 'dtype') and hasattr(df[col].dtype, 'byteorder'):
        if df[col].dtype.byteorder == '>':
            print(f"Converting {col} from big-endian to little-endian")
            df[col] = df[col].values.astype(df[col].dtype.newbyteorder('<'))

# data = df

print("DONE")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
data4 = pd.Series(data["APOGEE2_TARGET2"])

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# IDS = []
# 
# toUse = []
# flags = [
#     'APOGEE_ID', 'APOGEE2_TARGET2'
# ]
# features = toUse + flags
# 
# 
# df = pd.DataFrame()
# features=[] + IDS + features
# 
# features = list(set(features))
# 
# for feature in features:
#     print(f"Adding {feature}...", end=" ")
#     df[feature] = data[feature]
#     print("Done")

# === AFTER (edited) ===
IDS = []

toUse = []
flags = [
    'APOGEE_ID', 'APOGEE2_TARGET2'
]
features = toUse + flags


df = pd.DataFrame()
features=[] + IDS + features

features = list(set(features))

for feature in features:
    print(f"Adding {feature}...", end=" ")
    df[feature] = data[feature]
    # Convert big-endian FITS columns to native-endian for pandas ops
    if hasattr(df[feature].dtype, 'byteorder') and df[feature].dtype.byteorder == '>':
        df[feature] = df[feature].values.astype(df[feature].dtype.newbyteorder('<'))
    print("Done")

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
data2 = pd.read_csv("data/Orbits.csv")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# data3 = df.merge(data2, on='APOGEE_ID', how='right')

# === AFTER (edited) ===
# Ensure merge key columns are native-endian / string-compatible before merge
if hasattr(df['APOGEE_ID'].dtype, 'byteorder') and df['APOGEE_ID'].dtype.byteorder == '>':
    df['APOGEE_ID'] = df['APOGEE_ID'].values.astype(df['APOGEE_ID'].dtype.newbyteorder('<'))
if hasattr(data2['APOGEE_ID'].dtype, 'byteorder') and data2['APOGEE_ID'].dtype.byteorder == '>':
    data2['APOGEE_ID'] = data2['APOGEE_ID'].values.astype(data2['APOGEE_ID'].dtype.newbyteorder('<'))

df['APOGEE_ID'] = df['APOGEE_ID'].astype(str)
data2['APOGEE_ID'] = data2['APOGEE_ID'].astype(str)

data3 = df.merge(data2, on='APOGEE_ID', how='right')