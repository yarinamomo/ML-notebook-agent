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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
data2 = pd.read_csv("data/Orbits.csv")

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# data3 = df.merge(data2, on='APOGEE_ID', how='right')

# === AFTER (edited) ===
# Check if merge columns exist and have compatible types
merge_key = 'APOGEE_ID'

print(f"Columns in df: {list(df.columns)}")
print(f"Columns in data2: {list(data2.columns)}")

# Ensure merge key exists in both DataFrames
if merge_key not in df.columns:
    raise KeyError(f"'{merge_key}' not found in df")
if merge_key not in data2.columns:
    raise KeyError(f"'{merge_key}' not found in data2")

# Ensure the key types are compatible (convert to string if needed)
df[merge_key] = df[merge_key].astype(str)
data2[merge_key] = data2[merge_key].astype(str)

# Perform the merge
data3 = df.merge(data2, on=merge_key, how='right')

print("Merge completed successfully!")
print(f"Result shape: {data3.shape}")