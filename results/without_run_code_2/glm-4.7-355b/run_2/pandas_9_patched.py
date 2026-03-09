# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df = pd.read_csv("data/IMDb_All_Genres_etf_clean1.csv")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# clean_df = df[(df["Total Gross (millions)"]!="$0.00M") & (df["Total Gross (millions)"]!="Gross Unkown")].copy()
# clean_df = clean_df[clean_df['Censor']!="(Banned)"]

# === AFTER (edited) ===
# Try multiple possible column name variations for "Total Gross (millions)"
# Common variations include different spacing, case, or missing parentheses
possible_gross_columns = [
    "Total Gross (millions)",
    "Total Gross",
    "total_gross",
    "Total_Gross",
    "gross",
    "Gross",
    "Total Gross(millions)",
]

gross_col = None
for col_name in possible_gross_columns:
    if col_name in df.columns:
        gross_col = col_name
        print(f"Found column: '{col_name}'")
        break

if gross_col is None:
    # Print all columns for debugging
    print("Available columns:")
    for col in df.columns:
        print(f"  '{col}'")
    raise KeyError(f"Could not find any of these columns: {possible_gross_columns}")

# Filter based on the found column
clean_df = df[(df[gross_col]!="$0.00M") & (df[gross_col]!="Gross Unkown")].copy()

# Check if Censor column exists, otherwise try variations
if 'Censor' in df.columns:
    censor_col = 'Censor'
elif 'censor' in df.columns:
    censor_col = 'censor'
else:
    censor_col = None

if censor_col is not None:
    clean_df = clean_df[clean_df[censor_col]!="(Banned)"]
else:
    print("Warning: 'Censor' column not found, skipping that filter")
    print("Available columns:", df.columns.tolist())