# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import *
from sklearn.model_selection import *

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv("data/penguins.csv")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
df=df.dropna()

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df["sex"] = df["sex"].map({"male" : 0,"female" : 1})

# === AFTER (edited) ===
# Find the sex column (case-insensitive check)
sex_col = None
for col in df.columns:
    if col.lower().strip() == 'sex':
        sex_col = col
        break

if sex_col:
    df[sex_col] = df[sex_col].map({"male" : 0,"female" : 1})
else:
    print("Warning: 'sex' column not found in dataframe")

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 5}
sns.violinplot(df, x='species')