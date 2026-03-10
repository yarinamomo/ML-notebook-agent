# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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
# execution_status: {'status': 'not run'}
df = pd.read_csv("data/penguins.csv")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df=df.dropna()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df["sex"] = df["sex"].map({"male" : 0,"female" : 1})

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# sns.violinplot(df, x='species')

# === AFTER (edited) ===
sns.violinplot(df, x='species', y='bill_length_mm')