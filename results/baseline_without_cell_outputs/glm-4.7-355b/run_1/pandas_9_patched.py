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
# Let's first check what columns are available in the DataFrame
print("Available columns in the DataFrame:")
print(df.columns.tolist())
print("\n")

# Try multiple possible column name variations for Total Gross
gross_columns = [col for col in df.columns if 'Gross' in col.lower()]
print("Columns containing 'Gross':")
print(gross_columns)

# Use the first Gross-related column if available, or use a try-except approach
if gross_columns:
    gross_col = gross_columns[0]
    print(f"\nUsing column: '{gross_col}'")
    
    # Filter out rows with "$0.00M" or "Gross Unkown" values (adapt spelling variations)
    clean_df = df[(df[gross_col]!="$0.00M") & (df[gross_col]!="Gross Unkown") & (df[gross_col]!="Gross Unknown")].copy()
    
    # Filter out banned movies
    if 'Censor' in df.columns:
        clean_df = clean_df[clean_df['Censor']!="(Banned)"]
    elif 'Censor ' in df.columns:  # Check for trailing space
        clean_df = clean_df[clean_df['Censor ']!="(Banned)"]
    
    print(f"\nFiltered dataframe shape: {clean_df.shape}")
else:
    print("No 'Gross' column found. Showing first 5 rows:")
    print(df.head())
    clean_df = df.copy()  # Keep original if no Gross column found