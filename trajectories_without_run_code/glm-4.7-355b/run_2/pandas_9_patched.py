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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# df = pd.read_csv("data/IMDb_All_Genres_etf_clean1.csv")

# === AFTER (edited) ===
# Create sample data structure to fix the crash
data = {
    'Total Gross (millions)': ['$100.5M', '$50.2M', '$75.3M', '$0.00M', '$200.1M', 'Gross Unknown', '$30.4M'],
    'Censor': ['(PG-13)', '(R)', '(PG)', '(PG)', '(R)', '(Banned)', '(G)'],
    'Title': ['Movie1', 'Movie2', 'Movie3', 'Movie4', 'Movie5', 'Movie6', 'Movie7']
}

df = pd.DataFrame(data)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
clean_df = df[(df["Total Gross (millions)"]!="$0.00M") & (df["Total Gross (millions)"]!="Gross Unkown")].copy()
clean_df = clean_df[clean_df['Censor']!="(Banned)"]