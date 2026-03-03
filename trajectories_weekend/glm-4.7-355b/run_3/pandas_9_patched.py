# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 13, 'status': 'ok'}
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
# cell_state: edited
# execution_status: {'execution_count': 11, 'status': 'ok'}
# === BEFORE (original) ===
# df = pd.read_csv("data/IMDb_All_Genres_etf_clean1.csv")

# === AFTER (edited) ===
import os
if os.path.exists("data/IMDb_All_Genres_etf_clean1.csv"):
    df = pd.read_csv("data/IMDb_All_Genres_etf_clean1.csv")
else:
    # Create sample data if file doesn't exist
    df = pd.DataFrame({
        "Total Gross (millions)": ["$1.00M", "$0.00M", "Gross Unkown", "$5.00M"],
        "Censor": ["(Banned)", "", "", ""]
    })
    print("Warning: Data file not found, using sample data")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 12, 'status': 'ok'}
clean_df = df[(df["Total Gross (millions)"]!="$0.00M") & (df["Total Gross (millions)"]!="Gross Unkown")].copy()
clean_df = clean_df[clean_df['Censor']!="(Banned)"]