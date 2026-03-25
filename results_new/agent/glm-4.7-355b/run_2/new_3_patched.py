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
for dirname, _, filenames in os.walk('data/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
data = pd.read_excel('data/supermarket_sales.xlsx')
data.tail()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
data.head()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
data.info()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
data.describe(include='object')

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
data.describe()

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
data.nunique()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
data['datetime'] = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str), format='mixed')
data[['Date', 'Time', 'datetime']].head()

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
data['month'] = data['datetime'].dt.month
data['date'] = data['datetime'].dt.day
# data['hour'] = data['datetime'].dt.hour

# https://www.statology.org/pandas-day-of-week/
data['day_of_week'] = data['datetime'].dt.weekday
data['day_names'] = data['datetime'].dt.day_name()

data[['Date', 'Time','datetime', 'month', 'date', 'day_of_week', 'day_names']].head()

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
data[['Date', 'Time', 'datetime', 'month', 'date', 'day_of_week', 'day_names']].nunique()

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
duplicated_rows = data[data.duplicated()]
duplicated_rows

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
data.info()

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
index = data.columns
index

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
data.sort_values(['Branch', 'Total']).head().T

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
def describe_categorical_values(df):
    for col in df.columns:
        print(f"\nAttribute: {col}")
        
        # Extract unique values
        unique_vals = df[col].dropna().unique()
        
        # Sort values (works for strings, numbers, categoricals)
        try:
            sorted_vals = np.sort(unique_vals)
        except:
            # fallback for mixed types
            sorted_vals = sorted(unique_vals, key=lambda x: str(x))
        
        print("Unique values (sorted):")
        for v in sorted_vals:
            print(f"  - {v}")

cat_variables = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment', 'day_names']
describe_categorical_values(data[cat_variables])

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_all_categorical_pairs(df):
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Loop through columns in pairs
    for i in range(0, len(cat_cols), 2):
        col1 = cat_cols[i]
        col2 = cat_cols[i+1] if i+1 < len(cat_cols) else None

        # Create subplot layout
        if col2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(7, 5))
            axes = [axes]

        # Plot first column
        sns.countplot(
            x=df[col1],
            ax=axes[0],
            order=sorted(df[col1].dropna().unique())
        )
        axes[0].set_title(f"Distribution of {col1}")
        axes[0].tick_params(axis='x', rotation=45)

        # Plot second column if exists
        if col2:
            sns.countplot(
                x=df[col2],
                ax=axes[1],
                order=sorted(df[col2].dropna().unique())
            )
            axes[1].set_title(f"Distribution of {col2}")
            axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

plot_all_categorical_pairs(data[cat_variables])

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_analysis(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if numeric_df.empty:
        print("No numeric columns available for correlation analysis.")
        return

    # Compute correlation matrix
    corr_matrix = numeric_df.corr()

    print("Correlation Matrix:")
    print(corr_matrix)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


correlation_analysis(data)

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
print(f'Number of unique value in gross marging percentage: ', data['gross margin percentage'].nunique())
print(f'Unique value for gross margin percentage: ', data['gross margin percentage'].unique())

#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def cramers_v_matrix(df):
    # Select categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    n = len(cat_cols)

    # Create empty matrix
    result = pd.DataFrame(np.zeros((n, n)), index=cat_cols, columns=cat_cols)

    # Compute Cramér's V for each pair
    for col1 in cat_cols:
        for col2 in cat_cols:
            confusion_matrix = pd.crosstab(df[col1], df[col2])
            result.loc[col1, col2] = cramers_v(confusion_matrix)

    return result

cramers_v_corr = cramers_v_matrix(data)
print(cramers_v_corr)

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 20}
# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cramers_v_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Cramér's V Heatmap")
plt.tight_layout()
plt.show()

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 21}
import duckdb as dd

duck_df = dd.sql("SELECT * FROM data")

#%%
# --- [CELL 21]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 22}
dd.sql("SELECT * FROM duck_df TABLESAMPLE 1 ROWS;").show()

#%%
# --- [CELL 22]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 23}
# === BEFORE (original) ===
# import numpy as np
# 
# def correlation_ratio(categories, values):
#     # Remove missing values
#     mask = ~np.isnan(values)
#     categories = categories[mask]
#     values = values[mask]
# 
#     # Overall mean
#     grand_mean = values.mean()
# 
#     # Between-group variability
#     ss_between = 0
#     for cat in np.unique(categories):
#         group_vals = values[categories == cat]
#         ss_between += len(group_vals) * (group_vals.mean() - grand_mean) ** 2
# 
#     # Total variability
#     ss_total = ((values - grand_mean) ** 2).sum()
# 
#     # η = sqrt(SS_between / SS_total)
#     return np.sqrt(ss_between / ss_total) if ss_total != 0 else 0
# 
# 
# def eta_matrix(df):
#     cat_cols = df.select_dtypes(include=['object', 'category']).columns
#     num_cols = df.select_dtypes(include=['int64', 'float64']).columns
# 
#     result = pd.DataFrame(index=cat_cols, columns=num_cols, dtype=float)
# 
#     for cat in cat_cols:
#         for num in num_cols:
#             result.loc[cat, num] = correlation_ratio(df[cat].values, df[num].values)
# 
#     return result
# 
# 
# eta_corr = eta_matrix(data.loc[ : , data.columns != 'datetime'])
# print(eta_corr)

# === AFTER (edited) ===
import numpy as np

def correlation_ratio(categories, values):

    mask = ~np.isnan(values)
    categories = categories[mask]
    values = values[mask]

    grand_mean = values.mean()

    ss_between = 0
    # Convert categories to strings to handle mixed types (datetime, string, etc.)
    for cat in np.unique([str(c) for c in categories]):
        group_vals = values[[str(c) == cat for c in categories]]
        ss_between += len(group_vals) * (group_vals.mean() - grand_mean) ** 2

    ss_total = ((values - grand_mean) ** 2).sum()

    return np.sqrt(ss_between / ss_total) if ss_total != 0 else 0


def eta_matrix(df):
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    result = pd.DataFrame(index=cat_cols, columns=num_cols, dtype=float)

    for cat in cat_cols:
        for num in num_cols:
            result.loc[cat, num] = correlation_ratio(df[cat].values, df[num].values)

    return result


eta_corr = eta_matrix(data.loc[ : , data.columns != 'datetime'])
print(eta_corr)