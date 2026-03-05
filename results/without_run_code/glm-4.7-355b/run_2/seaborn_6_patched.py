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
if "sex" in df.columns:
    df["sex"] = df["sex"].map({"male" : 0,"female" : 1})

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# sns.violinplot(df, x='species')

# === AFTER (edited) ===
if 'species' in df.columns:
    # Check if there's a numeric column for y-axis
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        sns.violinplot(data=df, x='species', y=numeric_cols[0])