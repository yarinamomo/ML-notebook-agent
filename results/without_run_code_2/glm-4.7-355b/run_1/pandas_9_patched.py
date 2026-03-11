# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
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
# Find the correct column name by fuzzy matching
print("Columns containing 'gross' (case insensitive):")
gross_cols = [col for col in df.columns if 'gross' in col.lower()]
print(gross_cols)

print("\nColumns containing 'total' (case insensitive):")
total_cols = [col for col in df.columns if 'total' in col.lower()]
print(total_cols)

print("\nAll columns:")
print(df.columns.tolist())

# Try to use the best match column
if gross_cols:
    gross_col = gross_cols[0]
    print(f"\nUsing column: {gross_col}")
    clean_df = df[(df[gross_col]!="$0.00M") & (df[gross_col]!="Gross Unkown")].copy()
    clean_df = clean_df[clean_df['Censor']!="(Banned)"]
else:
    print("\nNo 'gross' column found. Using all data.")
    clean_df = df[df['Censor']!="(Banned)"].copy()