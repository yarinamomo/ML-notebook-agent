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
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# df["sex"] = df["sex"].map({"male" : 0,"female" : 1})

# === AFTER (edited) ===
# Check available columns and handle the sex column mapping
print("Available columns:", df.columns.tolist())

# Try to find the sex column with common variations
sex_column = None
for col in df.columns:
    if col.lower().strip() == 'sex':
        sex_column = col
        break

if sex_column:
    df["sex"] = df[sex_column].map({"male" : 0, "female" : 1})
else:
    print("Warning: 'sex' column not found. Skipping sex column mapping.")

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
sns.violinplot(df, x='species')