# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv("data/IMDb_All_Genres_etf_clean1.csv")

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# clean_df = df[(df["Total Gross (millions)"]!="$0.00M") & (df["Total Gross (millions)"]!="Gross Unkown")].copy()
# clean_df = clean_df[clean_df['Censor']!="(Banned)"]

# === AFTER (edited) ===
# First, let's see what columns are actually in the dataframe
print("Columns in df:", df.columns.tolist())

# Try to find similar column names
print("\nColumns containing 'Gross':", [col for col in df.columns if 'gross' in col.lower() or 'Gross' in col])
print("Columns containing 'Censor':", [col for col in df.columns if 'censor' in col.lower() or 'Censor' in col])

# Print first few rows to understand the data structure
print("\nFirst few rows:")
print(df.head())

# Now attempt filtering with the actual column names we found
# Based on typical column naming, let's try some common variations
clean_df = df.copy()

# Try different possible column names for Total Gross
gross_col = None
for col in df.columns:
    if 'gross' in col.lower():
        gross_col = col
        break

# Try different possible column names for Censor
censor_col = None
for col in df.columns:
    if 'censor' in col.lower():
        censor_col = col
        break

if gross_col and censor_col:
    clean_df = clean_df[(clean_df[gross_col] != "$0.00M") & (clean_df[gross_col] != "Gross Unkown")]
    clean_df = clean_df[clean_df[censor_col] != "(Banned)"]
    print(f"\nFiltered successfully using columns: '{gross_col}' and '{censor_col}'")
    print(f"Rows after filtering: {len(clean_df)}")
else:
    print(f"\nCould not find matching columns. Gross column: {gross_col}, Censor column: {censor_col}")