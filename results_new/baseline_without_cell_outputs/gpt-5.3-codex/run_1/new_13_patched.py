# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
#Import the required packages

import os
import calendar
from datetime import datetime
import math
import pandas as pd
import numpy as np

from IPython.display import display_markdown
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# reading the dataset
telecom = pd.read_csv("data/train.csv")

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# summary of the dataset:
telecom.info(verbose=True, show_counts=True)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# view the top 5 rows of the data

telecom.head()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
#function to plot data as table
def PlotAsTable(df, figName):
    print(f"----------------------------------------------------------------------------------------------------- \n Note: If you dont see the table '{figName}' below,\n please ensure the Jupyter Notebook is marked Trusted (File --> Trusted Notebook) \n-----------------------------------------------------------------------------------------------------")
    display_markdown(f'''#### {figName} ''',  raw=True)
    display_markdown("---",  raw=True)
    display_markdown(df.to_markdown(index = False), raw=True)
    #return df.to_markdown(index = False)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# read the data dictonary
data_dict = pd.read_csv("data/data_dictionary.csv")
PlotAsTable(data_dict, "Data Dictionary" )

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# Convert followng columns to datetime
# last_date_of_month_6, last_date_of_month_7, last_date_of_month_8,
# date_of_last_rech_6, date_of_last_rech_7, date_of_last_rech_8
# date_of_last_rech_data_6, date_of_last_rech_data_7, date_of_last_rech_data_8

dateColumns = telecom.select_dtypes(include='object')

#function that converts required columns to datetime
def ConvertDateTimeColumns(df):
    for dCol in dateColumns.columns:
        df[dCol] = pd.to_datetime(df[dCol])

#update the datetime columns
ConvertDateTimeColumns(telecom)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
#Rename Columns with Meaning full Names
# aug_vbc_3g jul_vbc_3g jun_vbc_3g

#function that renames required columns
def ApplyMeaningfulName(df):
    df.rename(columns={'jun_vbc_3g': 'vbc_3g_6', 'jul_vbc_3g': 'vbc_3g_7', 'aug_vbc_3g': 'vbc_3g_8'}, inplace=True)

ApplyMeaningfulName(telecom)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
def ShowSummary(df):
    df.info(verbose=True, show_counts=True)
# summary of the updated dataset:
ShowSummary(telecom)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
telecom.describe()

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
colDesrciption = []
data_dict['Acronyms'] = data_dict['Acronyms'].str.replace(" ","")
for colName in telecom.columns:
    colDescr = ""
    acronymDescription = data_dict.loc[data_dict['Acronyms'] == colName.upper()].head(1)
    if (not acronymDescription.empty):
        colDescr = f"{acronymDescription.iloc[0]['Description']}"
    else:
        for acronyms in colName.split("_"):
            acronymDescription = data_dict.loc[data_dict['Acronyms'] == acronyms.upper()].head(1)
            if (not acronymDescription.empty):
                colDescr = f" {colDescr} {acronyms}: {acronymDescription.iloc[0]['Description']},"
            elif (acronyms in ["6" , "7", "8"]):
                colDescr = f" {colDescr} {acronyms}: {calendar.month_name[int(acronyms)]},"
            elif (acronyms == "fb"):
                colDescr = f" {colDescr} fb_user: Service scheme to avail services of Facebook and similar social networking sites"
    colDesrciption.append(colDescr)

columnDescription = pd.DataFrame({'ColumnName': telecom.columns, 'Description':colDesrciption})
PlotAsTable(columnDescription, "Column Description" )

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# see the top 5 rows of the data
telecom.head()

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
def ReturnColumnsMissingPercentage(df):
    missing = df.isnull().sum() * 100 / len(df)
    missing.name = "MissingPercentage"
    missing = missing.to_frame().reset_index()
    return missing

percent_missing = ReturnColumnsMissingPercentage(telecom)
percent_missing["Description"] = columnDescription["Description"]
PlotAsTable(percent_missing.sort_values(by=['MissingPercentage'], ascending = False), "Missing Percentage")

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
def GetHighMissingCols(df):
    high_missing = df.loc[(df["MissingPercentage"] > 70.0)]["index"]
    return high_missing

high_missing_cols = GetHighMissingCols(percent_missing)
telecom.loc[:,high_missing_cols].describe()

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
recharge_cols_missing = ['total_rech_data_6','total_rech_data_7','total_rech_data_8',
                         'av_rech_amt_data_6','av_rech_amt_data_7','av_rech_amt_data_8']
def ReplaceRechargeColsForNoRecharge(df):
    for col in recharge_cols_missing:
        df[col] = df[col].replace(np.NaN,0.0)


ReplaceRechargeColsForNoRecharge(telecom)
ShowSummary(telecom)

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
# Calclate new column total recharge amount for data:  av_rech_amt_data * total_rech_data

#function that create total recharge amount data columns
def CreateTotalRechAmtDataCols(df):
    df['total_rech_amt_data_6'] = df.av_rech_amt_data_6 * df.total_rech_data_6
    df['total_rech_amt_data_7'] = df.av_rech_amt_data_7 * df.total_rech_data_7
    df['total_rech_amt_data_8'] = df.av_rech_amt_data_8 * df.total_rech_data_8

CreateTotalRechAmtDataCols(telecom)
telecom.describe()

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
print(telecom.quantile(np.arange(0.9, 1.01, 0.01)))

#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
#c
quantile_series = telecom.quantile(np.arange(0.9, 1.01, 0.01))
print(type(quantile_series))
print(quantile_series.index)
print(quantile_series.head())

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 20}
#c
quantile_series.index = range(len(quantile_series))  # Replace with integer index

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 21}
#c
quantile_series.index = quantile_series.index.astype(str)

#%%
# --- [CELL 21]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 22}
# === BEFORE (original) ===
# #c
# percentage_change = quantile_series.pct_change().mul(100)

# === AFTER (edited) ===
# Compute percentage change only on numeric columns to avoid datetime division errors
quantile_numeric = quantile_series.select_dtypes(include=[np.number])
percentage_change = quantile_numeric.pct_change().mul(100)