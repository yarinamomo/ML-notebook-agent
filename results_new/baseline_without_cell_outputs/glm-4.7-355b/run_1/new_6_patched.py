# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve)
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
import gc
from typing import List, Tuple, Optional
# Import Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

import optuna
import numpy as np
from xgboost import XGBClassifier
# from optuna.integration import XGBoostPruningCallback
from sklearn.metrics import roc_auc_score

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# Mixed Precision training (Tối ưu cho T4/P100)
from torch.cuda.amp import GradScaler, autocast

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# Base directory for the data
base_path = 'data/'

# 1. Main Application Data
app_train = pd.read_csv(os.path.join(base_path, 'application_train.csv'))
print('Training data shape: ', app_train.shape)

app_test = pd.read_csv(os.path.join(base_path, 'application_test.csv'))
print('Testing data shape: ', app_test.shape)

# 2. Supplementary Data
bureau = pd.read_csv(os.path.join(base_path, 'bureau.csv'))
print('Bureau data shape: ', bureau.shape)

pos_cash = pd.read_csv(os.path.join(base_path, 'POS_CASH_balance.csv'))
print('POS CASH balance shape: ', pos_cash.shape)

bureau_balance = pd.read_csv(os.path.join(base_path, 'bureau_balance.csv'))
print('Bureau balance shape: ', bureau_balance.shape)

credit_card = pd.read_csv(os.path.join(base_path, 'credit_card_balance.csv'))
print('Credit card balance shape: ', credit_card.shape)

installments = pd.read_csv(os.path.join(base_path, 'installments_payments.csv'))
print('Installments payments shape: ', installments.shape)

prev_app = pd.read_csv(os.path.join(base_path, 'previous_application.csv'))
print('Previous application shape: ', prev_app.shape)

# ---------change for reproducing purposes----------
app_train = app_train.sample(frac=0.1, random_state=42)
app_train = app_train.reset_index(drop=True)
app_test = app_test.sample(frac=0.1, random_state=42)
app_test = app_test.reset_index(drop=True)
bureau = bureau.sample(frac=0.1, random_state=42)
bureau = bureau.reset_index(drop=True)
pos_cash = pos_cash.sample(frac=0.1, random_state=42)
pos_cash = pos_cash.reset_index(drop=True)
bureau_balance = bureau_balance.sample(frac=0.1, random_state=42)
bureau_balance = bureau_balance.reset_index(drop=True)
credit_card = credit_card.sample(frac=0.1, random_state=42)
credit_card = credit_card.reset_index(drop=True)
installments = installments.sample(frac=0.1, random_state=42)
installments = installments.reset_index(drop=True)
prev_app = prev_app.sample(frac=0.1, random_state=42)
prev_app = prev_app.reset_index(drop=True)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
def one_hot_encoder(df: pd.DataFrame, nan_as_category: bool = True) -> tuple[pd.DataFrame, list]:
    """
    One-hot encodes object columns.
    Returns the transformed dataframe and list of new columns.
    """
    original_cols = list(df.columns)
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    df = pd.get_dummies(df, columns=categorical_cols, dummy_na=nan_as_category)
    new_cols = [c for c in df.columns if c not in original_cols]
    
    return df, new_cols


def group(df: pd.DataFrame, prefix: str, aggregations: dict, by: str = 'SK_ID_CURR') -> pd.DataFrame:
    """
    Performs groupby aggregation and renames columns with a prefix.
    """
    df_agg = df.groupby(by).agg(aggregations)
    df_agg.columns = pd.Index([f"{prefix}{col}_{stat.upper()}" for col, stat in df_agg.columns])
    
    return df_agg.reset_index()


def group_and_merge(df_agg: pd.DataFrame, df_base: pd.DataFrame, prefix: str, 
                    aggregations: dict, by: str = 'SK_ID_CURR') -> pd.DataFrame:
    """
    Aggregates 'df_agg' and merges the result into 'df_base'.
    """
    agg_result = group(df_agg, prefix, aggregations, by=by)
    return df_base.merge(agg_result, on=by, how='left')


def do_sum(df: pd.DataFrame, group_cols: list, col_to_sum: str, new_name: str) -> pd.DataFrame:
    """
    Sums a specific column by group and merges it back to the original dataframe.
    """
    sum_df = (
        df[group_cols + [col_to_sum]]
        .groupby(group_cols)[col_to_sum]
        .sum()
        .reset_index()
        .rename(columns={col_to_sum: new_name})
    )
    return df.merge(sum_df, on=group_cols, how='left')


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates through all columns and modifies the data type to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            
            # Compress Integer types
            if str(col_type).startswith('int'):
                # Check ranges for int8, int16, int32
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)

            # Compress Float types
            elif str(col_type).startswith('float'):
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% decreased)")
    return df

def lightgbm_feature_selection(df: pd.DataFrame, excluded_cols: list, auc_limit: float = 0.7) -> pd.DataFrame:
    """
    Recursive feature elimination using LightGBM Feature Importance.
    Removes features iteratively until AUC drops below limit.
    """
    # Clean column names for LightGBM compatibility
    df = df.rename(columns=lambda x: re.sub(r'[^A-Za-z0-9_]+', '_', x))
    
    clf = LGBMClassifier(random_state=0)
    
    # Split data (using only rows with valid targets)
    train_mask = df['TARGET'].notnull()
    X = df.loc[train_mask].drop('TARGET', axis=1)
    y = df.loc[train_mask, 'TARGET']
    
    # Initial feature set
    active_features = [c for c in X.columns if c not in excluded_cols]
    best_features = []
    max_auc = 1.0

    while max_auc > auc_limit:
        # Refine active features: Remove those kept in previous 'best_features' to narrow search
        # Note: Logic preserved from original request, though aggressive.
        if best_features:
            active_features = [f for f in active_features if f not in best_features]
        
        if not active_features: 
            break

        clf.fit(X[active_features], y)
        preds = clf.predict_proba(X[active_features])[:, 1]
        max_auc = roc_auc_score(y, preds)
        
        # Get important features (>0 importance)
        importances = pd.Series(clf.feature_importances_, index=active_features)
        best_features = importances[importances > 0].index.tolist()
        
        print(f"Current AUC: {max_auc:.4f} | Features kept: {len(best_features)}")

    # Drop the columns that remained in the last iteration (considered weak)
    df.drop(columns=active_features, axis=1, inplace=True, errors='ignore')
    
    return df

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
def application():
    """
    Build the main application dataset by merging train and test,
    performing cleaning, encoding, and extensive feature engineering.

    Returns:
        pd.DataFrame: Final processed application dataframe.
    """

    # ------------------------------------------------------------------
    # Merge train and test
    # ------------------------------------------------------------------
    df = app_train
    test_df = app_test
    df = pd.concat([df, test_df], axis=0).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Basic cleaning
    # ------------------------------------------------------------------
    # Remove unknown gender rows
    df = df[df["CODE_GENDER"] != "XNA"]

    # Remove extreme income outlier
    df = df[df["AMT_INCOME_TOTAL"] < 20_000_000]

    # Replace special missing values
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan, inplace=True)

    # ------------------------------------------------------------------
    # Binary categorical encoding
    # ------------------------------------------------------------------
    binary_features = ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

    for feature in binary_features:
        df[feature], _ = pd.factorize(df[feature])

    # ------------------------------------------------------------------
    # One-hot encoding for remaining categorical features
    # ------------------------------------------------------------------
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # ------------------------------------------------------------------
    # Document-related features
    # ------------------------------------------------------------------
    docs = [col for col in df.columns if "FLAG_DOC" in col]

    # Number of documents provided
    df["DOCUMENT_COUNT"] = df[docs].sum(axis=1)

    # Kurtosis of document flags
    df["NEW_DOC_KURT"] = df[docs].kurtosis(axis=1)

    # ------------------------------------------------------------------
    # Age grouping
    # ------------------------------------------------------------------
    def get_age_label(days_birth):
        """Return integer age group label."""
        age_years = -days_birth / 365

        if age_years < 27:
            return 1
        elif age_years < 40:
            return 2
        elif age_years < 50:
            return 3
        elif age_years < 65:
            return 4
        elif age_years < 99:
            return 5
        return 0

    df["AGE_RANGE"] = df["DAYS_BIRTH"].apply(get_age_label)

    # ------------------------------------------------------------------
    # External source features
    # ------------------------------------------------------------------
    df["EXT_SOURCES_PROD"] = (
        df["EXT_SOURCE_1"]
        * df["EXT_SOURCE_2"]
        * df["EXT_SOURCE_3"]
    )

    df["EXT_SOURCES_WEIGHTED"] = (
        df["EXT_SOURCE_1"] * 2
        + df["EXT_SOURCE_2"]
        + df["EXT_SOURCE_3"] * 3
    )

    warnings.filterwarnings(
        "ignore",
        r"All-NaN (slice|axis) encountered"
    )

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

    for func in ["min", "max", "mean", "nanmedian", "var"]:
        feature_name = f"EXT_SOURCES_{func.upper()}"

        df[feature_name] = getattr(np, func)(
            df[ext_cols],
            axis=1
        )

    # ------------------------------------------------------------------
    # Basic ratio features
    # ------------------------------------------------------------------
    df["DAYS_EMPLOYED_PERC"] = (
        df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    )

    df["INCOME_CREDIT_PERC"] = (
        df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    )

    df["INCOME_PER_PERSON"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    )

    df["ANNUITY_INCOME_PERC"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    )

    df["PAYMENT_RATE"] = (
        df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    )

    # ------------------------------------------------------------------
    # Credit and income ratios
    # ------------------------------------------------------------------
    df["CREDIT_TO_GOODS_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    )

    df["INCOME_TO_EMPLOYED_RATIO"] = (
        df["AMT_INCOME_TOTAL"] / df["DAYS_EMPLOYED"]
    )

    df["INCOME_TO_BIRTH_RATIO"] = (
        df["AMT_INCOME_TOTAL"] / df["DAYS_BIRTH"]
    )

    # ------------------------------------------------------------------
    # Time-based ratios
    # ------------------------------------------------------------------
    df["ID_TO_BIRTH_RATIO"] = (
        df["DAYS_ID_PUBLISH"] / df["DAYS_BIRTH"]
    )

    df["CAR_TO_BIRTH_RATIO"] = (
        df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
    )

    df["CAR_TO_EMPLOYED_RATIO"] = (
        df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"]
    )

    df["PHONE_TO_BIRTH_RATIO"] = (
        df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    )

    # ------------------------------------------------------------------
    # Aggregated EXT_SOURCE statistics
    # ------------------------------------------------------------------
    df["APPS_EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

    df["APPS_EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["APPS_EXT_SOURCE_STD"].fillna(
        df["APPS_EXT_SOURCE_STD"].mean(),
        inplace=True
    )

    # Score-to-age ratios
    df["APP_SCORE1_TO_BIRTH_RATIO"] = (
        df["EXT_SOURCE_1"] / (df["DAYS_BIRTH"] / 365.25)
    )
    df["APP_SCORE2_TO_BIRTH_RATIO"] = (
        df["EXT_SOURCE_2"] / (df["DAYS_BIRTH"] / 365.25)
    )
    df["APP_SCORE3_TO_BIRTH_RATIO"] = (
        df["EXT_SOURCE_3"] / (df["DAYS_BIRTH"] / 365.25)
    )

    # Score-to-employment ratios
    df["APP_SCORE1_TO_EMPLOY_RATIO"] = (
        df["EXT_SOURCE_1"] / (df["DAYS_EMPLOYED"] / 365.25)
    )

    # Interaction features
    df["APP_EXT_SOURCE_2*EXT_SOURCE_3*DAYS_BIRTH"] = (
        df["EXT_SOURCE_1"]
        * df["EXT_SOURCE_2"]
        * df["DAYS_BIRTH"]
    )

    df["APP_SCORE1_TO_FAM_CNT_RATIO"] = (
        df["EXT_SOURCE_1"] / df["CNT_FAM_MEMBERS"]
    )

    df["APP_SCORE1_TO_GOODS_RATIO"] = (
        df["EXT_SOURCE_1"] / df["AMT_GOODS_PRICE"]
    )

    df["APP_SCORE1_TO_CREDIT_RATIO"] = (
        df["EXT_SOURCE_1"] / df["AMT_CREDIT"]
    )

    df["APP_SCORE1_TO_SCORE2_RATIO"] = (
        df["EXT_SOURCE_1"] / df["EXT_SOURCE_2"]
    )

    df["APP_SCORE1_TO_SCORE3_RATIO"] = (
        df["EXT_SOURCE_1"] / df["EXT_SOURCE_3"]
    )

    df["APP_SCORE2_TO_CREDIT_RATIO"] = (
        df["EXT_SOURCE_2"] / df["AMT_CREDIT"]
    )

    df["APP_SCORE2_TO_REGION_RATING_RATIO"] = (
        df["EXT_SOURCE_2"] / df["REGION_RATING_CLIENT"]
    )

    df["APP_SCORE2_TO_CITY_RATING_RATIO"] = (
        df["EXT_SOURCE_2"]
        / df["REGION_RATING_CLIENT_W_CITY"]
    )

    df["APP_SCORE2_TO_POP_RATIO"] = (
        df["EXT_SOURCE_2"]
        / df["REGION_POPULATION_RELATIVE"]
    )

    df["APP_SCORE2_TO_PHONE_CHANGE_RATIO"] = (
        df["EXT_SOURCE_2"]
        / df["DAYS_LAST_PHONE_CHANGE"]
    )

    # Additional interactions
    df["APP_EXT_SOURCE_1*EXT_SOURCE_2"] = (
        df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"]
    )
    df["APP_EXT_SOURCE_1*EXT_SOURCE_3"] = (
        df["EXT_SOURCE_1"] * df["EXT_SOURCE_3"]
    )
    df["APP_EXT_SOURCE_2*EXT_SOURCE_3"] = (
        df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    )

    df["APP_EXT_SOURCE_1*DAYS_EMPLOYED"] = (
        df["EXT_SOURCE_1"] * df["DAYS_EMPLOYED"]
    )
    df["APP_EXT_SOURCE_2*DAYS_EMPLOYED"] = (
        df["EXT_SOURCE_2"] * df["DAYS_EMPLOYED"]
    )
    df["APP_EXT_SOURCE_3*DAYS_EMPLOYED"] = (
        df["EXT_SOURCE_3"] * df["DAYS_EMPLOYED"]
    )

    # ------------------------------------------------------------------
    # Income-related ratios
    # ------------------------------------------------------------------
    df["APPS_GOODS_INCOME_RATIO"] = (
        df["AMT_GOODS_PRICE"] / df["AMT_INCOME_TOTAL"]
    )

    df["APPS_CNT_FAM_INCOME_RATIO"] = (
        df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    )

    df["APPS_INCOME_EMPLOYED_RATIO"] = (
        df["AMT_INCOME_TOTAL"] / df["DAYS_EMPLOYED"]
    )

    # ------------------------------------------------------------------
    # Additional engineered features
    # ------------------------------------------------------------------
    df["CREDIT_TO_GOODS_RATIO_2"] = (
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    )

    df["APP_AMT_INCOME_TOTAL_12_AMT_ANNUITY_ratio"] = (
        df["AMT_INCOME_TOTAL"] / 12.0 - df["AMT_ANNUITY"]
    )

    df["APP_INCOME_TO_EMPLOYED_RATIO"] = (
        df["AMT_INCOME_TOTAL"] / df["DAYS_EMPLOYED"]
    )

    df["APP_DAYS_LAST_PHONE_CHANGE_DAYS_EMPLOYED_ratio"] = (
        df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"]
    )

    df["APP_DAYS_EMPLOYED_DAYS_BIRTH_diff"] = (
        df["DAYS_EMPLOYED"] - df["DAYS_BIRTH"]
    )

    print('"Application_Train_Test" final shape:', df.shape)

    return df

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
def bureau_bb(bureau=bureau, bureau_balance=bureau_balance):
    bb = bureau_balance

    # Credit duration and credit/account end date difference
    bureau['CREDIT_DURATION'] = -bureau['DAYS_CREDIT'] + bureau['DAYS_CREDIT_ENDDATE']
    bureau['ENDDATE_DIF'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
    
    # Credit to debt ratio and difference
    bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']
    bureau['BUREAU_CREDIT_FACT_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
    bureau['BUREAU_CREDIT_ENDDATE_DIFF'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['BUREAU_CREDIT_DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']

    # CREDIT_DAY_OVERDUE :
    bureau['BUREAU_IS_DPD'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 0 else 0)
    bureau['BUREAU_IS_DPD_OVER120'] = bureau['CREDIT_DAY_OVERDUE'].apply(lambda x: 1 if x > 120 else 0)

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size', 'mean']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']

    #Status of Credit Bureau loan during the month
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'min'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max', 'sum'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
        'SK_ID_BUREAU': ['count'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'ENDDATE_DIF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_FACT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_ENDDATE_DIFF': ['min', 'max', 'mean'],
        'BUREAU_CREDIT_DEBT_RATIO': ['min', 'max', 'mean'],
        'DEBT_CREDIT_DIFF': ['min', 'max', 'mean'],
        'BUREAU_IS_DPD': ['mean', 'sum'],
        'BUREAU_IS_DPD_OVER120': ['mean', 'sum']
        }

    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')

    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')

    print('"Bureau/Bureau Balance" final shape:', bureau_agg.shape)
    return bureau_agg

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
def previous_application():
    prev = prev_app

    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)

    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']

    # Feature engineering: ratios and difference
    prev['APPLICATION_CREDIT_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_CREDIT']
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']

    # Interest ratio on previous application (simplified)
    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']

    # Days last due difference (scheduled x done)
    prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

    # from off
    prev['PREV_GOODS_DIFF'] = prev['AMT_APPLICATION'] - prev['AMT_GOODS_PRICE']
    prev['PREV_ANNUITY_APPL_RATIO'] = prev['AMT_ANNUITY']/prev['AMT_APPLICATION']
    prev['PREV_GOODS_APPL_RATIO'] = prev['AMT_GOODS_PRICE'] / prev['AMT_APPLICATION']

    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'SK_ID_PREV': ['nunique'],
        'DAYS_TERMINATION': ['max'],
        'CREDIT_TO_ANNUITY_RATIO': ['mean', 'max'],
        'APPLICATION_CREDIT_DIFF': ['min', 'max', 'mean', 'sum'],
        'DOWN_PAYMENT_TO_CREDIT': ['mean'],
        'PREV_GOODS_DIFF': ['mean', 'max', 'sum'],
        'PREV_GOODS_APPL_RATIO': ['mean', 'max'],
        'DAYS_LAST_DUE_DIFF': ['mean', 'max', 'sum'],
        'SIMPLE_INTERESTS': ['mean', 'max']
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')

    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')

    print('"Previous Applications" final shape:', prev_agg.shape)
    return prev_agg

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# Sửa dòng def: Truyền DataFrame pos_cash vào làm tham số mặc định (đặt tên là df cho gọn)
def pos_cash(df=pos_cash):
    pos = df # Gán pos bằng tham số df đã truyền vào

    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)

    # Flag months with late payment
    pos['LATE_PAYMENT'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    pos['POS_IS_DPD'] = pos['SK_DPD'].apply(lambda x: 1 if x > 0 else 0) # <-- same with ['LATE_PAYMENT']
    pos['POS_IS_DPD_UNDER_120'] = pos['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
    pos['POS_IS_DPD_OVER_120'] = pos['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size', 'min'],
        'SK_DPD': ['max', 'mean', 'sum', 'var', 'min'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        'SK_ID_PREV': ['nunique'],
        'LATE_PAYMENT': ['mean'],
        'SK_ID_CURR': ['count'],
        'CNT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean', 'sum'],
        'POS_IS_DPD': ['mean', 'sum'],
        'POS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'POS_IS_DPD_OVER_120': ['mean', 'sum'],
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])

    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()


    sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    gp = sort_pos.groupby('SK_ID_PREV')
    df_pos = pd.DataFrame()
    df_pos['SK_ID_CURR'] = gp['SK_ID_CURR'].first()
    df_pos['MONTHS_BALANCE_MAX'] = gp['MONTHS_BALANCE'].max()

    # Percentage of previous loans completed and completed before initial term
    df_pos['POS_LOAN_COMPLETED_MEAN'] = gp['NAME_CONTRACT_STATUS_Completed'].mean()
    df_pos['POS_COMPLETED_BEFORE_MEAN'] = gp['CNT_INSTALMENT'].first() - gp['CNT_INSTALMENT'].last()
    df_pos['POS_COMPLETED_BEFORE_MEAN'] = df_pos.apply(lambda x: 1 if x['POS_COMPLETED_BEFORE_MEAN'] > 0 \
                                                                      and x['POS_LOAN_COMPLETED_MEAN'] > 0 else 0, axis=1)
    # Number of remaining installments (future installments) and percentage from total
    df_pos['POS_REMAINING_INSTALMENTS'] = gp['CNT_INSTALMENT_FUTURE'].last()
    df_pos['POS_REMAINING_INSTALMENTS_RATIO'] = gp['CNT_INSTALMENT_FUTURE'].last()/gp['CNT_INSTALMENT'].last()

    # Group by SK_ID_CURR and merge
    df_gp = df_pos.groupby('SK_ID_CURR').sum().reset_index()
    df_gp.drop(['MONTHS_BALANCE_MAX'], axis=1, inplace= True)
    pos_agg = pd.merge(pos_agg, df_gp, on= 'SK_ID_CURR', how= 'left')

    # Percentage of late payments for the 3 most recent applications
    pos = do_sum(pos, ['SK_ID_PREV'], 'LATE_PAYMENT', 'LATE_PAYMENT_SUM')

    # Last month of each application
    last_month_df = pos.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()

    # Most recent applications (last 3)
    sort_pos = pos.sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    gp = sort_pos.iloc[last_month_df].groupby('SK_ID_CURR').tail(3)
    gp_mean = gp.groupby('SK_ID_CURR').mean().reset_index()
    pos_agg = pd.merge(pos_agg, gp_mean[['SK_ID_CURR', 'LATE_PAYMENT_SUM']], on='SK_ID_CURR', how='left')

    print('"Pos-Cash" balance final shape:', pos_agg.shape) 
    return pos_agg

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
def installment():
    ins = installments

    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)

    # Group payments and get Payment difference
    ins = do_sum(ins, ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'], 'AMT_PAYMENT', 'AMT_PAYMENT_GROUPED')
    ins['PAYMENT_DIFFERENCE'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT_GROUPED']
    ins['PAYMENT_RATIO'] = ins['AMT_INSTALMENT'] / ins['AMT_PAYMENT_GROUPED']
    ins['PAID_OVER_AMOUNT'] = ins['AMT_PAYMENT'] - ins['AMT_INSTALMENT']
    ins['PAID_OVER'] = (ins['PAID_OVER_AMOUNT'] > 0).astype(int)

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']

    # Days past due and days before due (no negative values)
    ins['DPD_diff'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD_diff'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD_diff'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD_diff'].apply(lambda x: x if x > 0 else 0)

    # Flag late payment
    ins['LATE_PAYMENT'] = ins['DBD'].apply(lambda x: 1 if x > 0 else 0)
    ins['INSTALMENT_PAYMENT_RATIO'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['LATE_PAYMENT_RATIO'] = ins.apply(lambda x: x['INSTALMENT_PAYMENT_RATIO'] if x['LATE_PAYMENT'] == 1 else 0, axis=1)

    # Flag late payments that have a significant amount
    ins['SIGNIFICANT_LATE_PAYMENT'] = ins['LATE_PAYMENT_RATIO'].apply(lambda x: 1 if x > 0.05 else 0)
    
    # Flag k threshold late payments
    ins['DPD_7'] = ins['DPD'].apply(lambda x: 1 if x >= 7 else 0)
    ins['DPD_15'] = ins['DPD'].apply(lambda x: 1 if x >= 15 else 0)

    ins['INS_IS_DPD_UNDER_120'] = ins['DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
    ins['INS_IS_DPD_OVER_120'] = ins['DPD'].apply(lambda x: 1 if (x >= 120) else 0)

    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'var'],
        'DBD': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'min'],
        'SK_ID_PREV': ['size', 'nunique'],
        'PAYMENT_DIFFERENCE': ['mean'],
        'PAYMENT_RATIO': ['mean', 'max'],
        'LATE_PAYMENT': ['mean', 'sum'],
        'SIGNIFICANT_LATE_PAYMENT': ['mean', 'sum'],
        'LATE_PAYMENT_RATIO': ['mean'],
        'DPD_7': ['mean'],
        'DPD_15': ['mean'],
        'PAID_OVER': ['mean'],
        'DPD_diff':['mean', 'min', 'max'],
        'DBD_diff':['mean', 'min', 'max'],
        'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
        'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'INS_IS_DPD_OVER_120': ['mean', 'sum']
    }

    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])

    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()

    # from oof (DAYS_ENTRY_PAYMENT)
    cond_day = ins['DAYS_ENTRY_PAYMENT'] >= -365
    ins_d365_grp = ins[cond_day].groupby('SK_ID_CURR')
    ins_d365_agg_dict = {
        'SK_ID_CURR': ['count'],
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_ENTRY_PAYMENT': ['mean', 'max', 'sum'],
        'DAYS_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_INSTALMENT': ['mean', 'max', 'sum'],
        'AMT_PAYMENT': ['mean', 'max', 'sum'],
        'PAYMENT_DIFF': ['mean', 'min', 'max', 'sum'],
        'PAYMENT_PERC': ['mean', 'max'],
        'DPD_diff': ['mean', 'min', 'max'],
        'DPD': ['mean', 'sum'],
        'INS_IS_DPD_UNDER_120': ['mean', 'sum'],
        'INS_IS_DPD_OVER_120': ['mean', 'sum']}

    ins_d365_agg = ins_d365_grp.agg(ins_d365_agg_dict)
    ins_d365_agg.columns = ['INS_D365' + ('_').join(column).upper() for column in ins_d365_agg.columns.ravel()]

    ins_agg = ins_agg.merge(ins_d365_agg, on='SK_ID_CURR', how='left')

    print('"Installments Payments" final shape:', ins_agg.shape)
    return ins_agg

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
def process_credit_card(cc_df=None):    
    if cc_df is None:
        cc = credit_card 
    else:
        cc = cc_df

    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    # Amount used from limit
    cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    # Current payment / Min payment
    cc['PAYMENT_DIV_MIN'] = cc['AMT_PAYMENT_CURRENT'] / cc['AMT_INST_MIN_REGULARITY']
    # Late payment <-- 'CARD_IS_DPD'
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    # How much drawing of limit
    cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

    cc['CARD_IS_DPD_UNDER_120'] = cc['SK_DPD'].apply(lambda x: 1 if (x > 0) & (x < 120) else 0)
    cc['CARD_IS_DPD_OVER_120'] = cc['SK_DPD'].apply(lambda x: 1 if x >= 120 else 0)

    # General aggregations
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    # Last month balance of each credit card application
    last_ids = cc.groupby('SK_ID_PREV')['MONTHS_BALANCE'].idxmax()
    last_months_df = cc[cc.index.isin(last_ids)]
    cc_agg = group_and_merge(last_months_df,cc_agg,'CC_LAST_', {'AMT_BALANCE': ['mean', 'max']})

    CREDIT_CARD_TIME_AGG = {
        'AMT_BALANCE': ['mean', 'max'],
        'LIMIT_USE': ['max', 'mean'],
        'AMT_CREDIT_LIMIT_ACTUAL':['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_CURRENT': ['max', 'sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['max', 'sum'],
        'AMT_INST_MIN_REGULARITY': ['max', 'mean'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max','sum'],
        'AMT_TOTAL_RECEIVABLE': ['max', 'mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max','sum', 'mean'],
        'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['mean'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'LIMIT_USE': ['min', 'max'],
        'DRAWING_LIMIT_RATIO': ['min', 'max'],
        'LATE_PAYMENT': ['mean', 'sum'],
        'CARD_IS_DPD_UNDER_120': ['mean', 'sum'],
        'CARD_IS_DPD_OVER_120': ['mean', 'sum']
    }

    for months in [12, 24, 48]:
        cc_prev_id = cc[cc['MONTHS_BALANCE'] >= -months]['SK_ID_PREV'].unique()
        cc_recent = cc[cc['SK_ID_PREV'].isin(cc_prev_id)]
        prefix = 'INS_{}M_'.format(months)
        cc_agg = group_and_merge(cc_recent, cc_agg, prefix, CREDIT_CARD_TIME_AGG)

    print('"Credit Card Balance" final shape:', cc_agg.shape)
    return cc_agg

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
nan_as_category = True
def risk_groupanizer(
    dataframe: pd.DataFrame, 
    column_names: List[str], 
    train_df: Optional[pd.DataFrame] = None, 
    target_col: str = 'TARGET', 
    target_val: int = 1, 
    upper_limit_ratio: float = 8.2, 
    lower_limit_ratio: float = 8.2
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tạo các feature mới dựa trên tỷ lệ rủi ro (Risk Ratio) của từng nhóm category.
    """
    # 1. Xác định dữ liệu dùng để học quy luật (tránh Data Leakage)
    calc_df = train_df if train_df is not None else dataframe
    
    if target_col not in calc_df.columns:
        raise ValueError(f"Cột Target '{target_col}' không tồn tại trong dữ liệu tính toán.")

    new_cols_created = []

    # Copy nhẹ các cột cần thiết để xử lý vụ float16 mà không ảnh hưởng DF gốc
    # Chỉ cần ID và Target để group
    base_cols = [target_col]
    
    # Ép kiểu an toàn cho Target trong calc_df cục bộ
    work_df = calc_df[base_cols].copy()
    if work_df[target_col].dtype == 'float16':
        work_df[target_col] = work_df[target_col].astype('float32')

    for col in column_names:
        if col not in dataframe.columns:
            continue
        
        # Thêm cột feature vào work_df để tính toán
        # Lưu ý: Nếu col không có trong calc_df (trường hợp lệch pha), skip
        if col not in calc_df.columns:
            continue
            
        current_series = calc_df[col]
        if current_series.dtype == 'float16':
            current_series = current_series.astype('float32')
        
        work_df[col] = current_series

        # --- BƯỚC 1: TÍNH TOÁN TỶ LỆ RỦI RO ---
        # Groupby: Đếm số lượng hồ sơ theo Feature Category và Target Value
        # count() dùng cột target_col hay cột bất kỳ đều được miễn là không null
        grp = work_df.groupby([col, target_col]).size().reset_index(name='count')
        
        # Tổng số lượng hồ sơ trong mỗi Category của Feature
        total_per_cat = grp.groupby(col)['count'].transform('sum')
        grp['ratio%'] = round(grp['count'] * 100 / total_per_cat, 1)

        # Lọc ra các nhóm thỏa mãn target_val (thường là 1 - nợ xấu)
        target_mask = grp[target_col] == target_val
        
        high_risk_groups = grp.loc[target_mask & (grp['ratio%'] >= upper_limit_ratio), col].tolist()
        low_risk_groups = grp.loc[target_mask & (grp['ratio%'] <= lower_limit_ratio), col].tolist()

        risk_mapping = {
            '_high_risk': high_risk_groups,
            '_low_risk': low_risk_groups
        }

        if upper_limit_ratio != lower_limit_ratio:
            med_mask = target_mask & (grp['ratio%'] < upper_limit_ratio) & (grp['ratio%'] > lower_limit_ratio)
            medium_risk_groups = grp.loc[med_mask, col].tolist()
            risk_mapping['_medium_risk'] = medium_risk_groups

        # --- BƯỚC 2: ÁP DỤNG VÀO DATAFRAME GỐC ---
        for suffix, groups in risk_mapping.items():
            if not groups: # Bỏ qua nếu danh sách rỗng
                continue
            new_col_name = f"{col}{suffix}"
            dataframe[new_col_name] = dataframe[col].isin(groups).astype('int8') # int8 cho nhẹ
            new_cols_created.append(new_col_name)

        # Drop cột gốc nếu là object/category sau khi đã encode
        if pd.api.types.is_object_dtype(dataframe[col]) or pd.api.types.is_categorical_dtype(dataframe[col]):
            dataframe.drop(col, axis=1, inplace=True)
        
        # Dọn dẹp cột tạm khỏi work_df để tiết kiệm mem cho vòng lặp sau
        work_df.drop(col, axis=1, inplace=True)

    return dataframe, new_cols_created

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
def data_post_processing(dataframe: pd.DataFrame) -> pd.DataFrame:
    print(f'---=> DATA POST-PROCESSING begins, dataset has {dataframe.shape[1]} features')

    # Index columns cần giữ lại, không xử lý
    index_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
    existing_index_cols = [c for c in index_cols if c in dataframe.columns]

    # 1. Rename columns (Chuẩn hóa tên cột: bỏ ký tự đặc biệt)
    dataframe = dataframe.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
    print('Feature names renamed')

    # 2. Reduce memory usage (Giả định hàm này đã có)
    if 'reduce_mem_usage' in globals():
        dataframe = reduce_mem_usage(dataframe)
        print('Memory usage reduced')

    # 3. Remove non-informative columns (Chỉ có 1 giá trị duy nhất)
    const_cols = [col for col in dataframe.columns if dataframe[col].nunique() < 2]
    if const_cols:
        dataframe.drop(columns=const_cols, inplace=True)
        print(f'Dropped {len(const_cols)} constant columns.')

    print(f'{dataframe.shape[1]} features remain after removing non-informative columns')

    # 4. Handle Object Columns for LightGBM
    print('Checking for object columns to fix LightGBM crash...')
    # Lấy danh sách cột object trừ các cột index/ID
    object_cols = [col for col in dataframe.columns 
                   if dataframe[col].dtype == 'object' and col not in existing_index_cols]

    for col in object_cols:
        # Thử chuyển sang số trước
        dataframe[col] = pd.to_numeric(dataframe[col], errors='ignore')
        
        # Nếu vẫn là object (tức là string thực sự), dùng Factorize
        if dataframe[col].dtype == 'object':
            dataframe[col], _ = pd.factorize(dataframe[col])
            
    print(f'Processed {len(object_cols)} object columns.')

    # 5. Feature Selection using LightGBM
    feature_num_before = dataframe.shape[1]
    auc_limit = 0.7
    
    if 'ligthgbm_feature_selection' in globals():
        dataframe = ligthgbm_feature_selection(
            dataframe,
            index_cols=existing_index_cols,
            auc_limit=auc_limit
        )
        print(f'{feature_num_before - dataframe.shape[1]} features eliminated by LightGBM')
    else:
        print("Warning: 'ligthgbm_feature_selection' function not found. Skipping selection.")
        
    print(f'{dataframe.shape[1]} features remain after feature selection')

    # 6. Generate new columns with risk_groupanizer
    start_feats_num = dataframe.shape[1]
    
    # Tìm các cột Categorical (dạng số int nhưng ít giá trị unique) để group risk
    # Điều kiện: 3 < unique values < 20 (heuristic)
    cat_cols = [col for col in dataframe.columns 
                if 3 < dataframe[col].nunique() < 20 
                and col not in existing_index_cols]
    
    dataframe, _ = risk_groupanizer(
        dataframe, 
        column_names=cat_cols, 
        upper_limit_ratio=8.1, 
        lower_limit_ratio=8.1
    )
    
    print(f'---=> {dataframe.shape[1] - start_feats_num} features generated with risk_groupanizer')
    print(f'---=> DATA POST-PROCESSING ended! Total features: {dataframe.shape[1]}')

    gc.collect()
    return dataframe

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
df = application()
df = df.merge(bureau_bb(), how='left', on='SK_ID_CURR')
print('--=> df after merge with bureau:', df.shape)
df = df.merge(previous_application(), how='left', on='SK_ID_CURR')
print('--=> df after merge with previous application:', df.shape)
df = df.merge(pos_cash(), how='left', on='SK_ID_CURR')
print('--=> df after merge with pos cash :', df.shape)
df = df.merge(installment(), how='left', on='SK_ID_CURR')
print('--=> df after merge with installments:', df.shape)
df = df.merge(process_credit_card(), how='left', on='SK_ID_CURR')
print('--=> df after merge with credit card:', df.shape)
df = data_post_processing(df)
print('='*50, '\n')
print('---=> df final shape:', df.shape, ' <=---', '\n')
print('=' * 50)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
import re
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Clean collumns name 
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

# 2. Train test split
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]

print(f"Train shape: {train_df.shape}")
print(f"Test (Submission) shape: {test_df.shape}")

drop_cols = ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']
feats = [f for f in train_df.columns if f not in drop_cols]

X = train_df[feats]
y = train_df['TARGET']
X_test = test_df[feats] 

# 3. Split Train/Validation (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_val: {X_test.shape}")

del df, train_df
gc.collect()

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
import numpy as np
import pandas as pd

print("Checking for infinity values...")
def replace_inf_with_nan(df):
    if isinstance(df, pd.DataFrame):
        n_inf = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
        print(f"Found {n_inf} infinite values. Replacing with NaN...")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Đảm bảo kiểu dữ liệu là float32 (để tiết kiệm RAM và tránh lỗi float16 nếu có
        # XGBoost đôi khi không thích float16
        for col in df.select_dtypes(include=['float16']).columns:
            df[col] = df[col].astype('float32')
            
    return df

# Áp dụng cho X_train và X_val
X_train = replace_inf_with_nan(X_train)
X_val = replace_inf_with_nan(X_val)

print("Data cleaning completed!")

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
if 'ratio' not in locals():
    y_train_np = np.array(y_train)
    ratio = float(np.sum(y_train_np == 0)) / np.sum(y_train_np == 1)
    print(f"Calculated Scale Pos Weight Ratio: {ratio:.4f}")

N_TRIAL_JOBS = 2
N_THREADS = 4
def objective(trial):
    param = {
        # --- BASIC ---
        'n_estimators': 2, #2000,# ---------change for reproducing purposes----------
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),

        # --- TREE ---
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),

        # --- REGULARIZATION ---
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),

        # --- CLASS BALANCE ---
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, ratio * 1.5),

        # --- SYSTEM ---
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'nthread': N_THREADS,
        'random_state': 42,
        'eval_metric': 'auc',
        'device': 'cuda', 
        # --- EARLY STOPPING ---
        'early_stopping_rounds': 50
    }

    model = XGBClassifier(**param)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    return auc

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
# Run Optuna
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42)
)

print("Optimizing XGBoost...")

study.optimize(
    objective,
    n_trials=2, #100, # ---------change for reproducing purposes----------
    n_jobs=N_TRIAL_JOBS,
    gc_after_trial=True
)

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
# Results
print("Best trial AUC:", study.best_value)
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Train Final Model
best_params = study.best_params
final_xgb_params = {
    **best_params,
    'n_estimators': 3, #3000, # ---------change for reproducing purposes----------
    'tree_method': 'hist',
    'objective': 'binary:logistic',
    'nthread': N_THREADS,
    'random_state': 42,
    'eval_metric': 'auc',
    'early_stopping_rounds': 100,
    'device': 'cuda'
}

print("Training Final XGBoost Model...")
final_xgb = XGBClassifier(**final_xgb_params)

final_xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

preds = final_xgb.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, preds)
print("Final XGB AUC:", auc)

#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
scaler = StandardScaler()
X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e5, neginf=-1e5)
X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e5, neginf=-1e5)
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
pos_weight_val = (y_train == 0).sum() / (y_train == 1).sum()
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 20}
class AdvancedFraudANN(nn.Module):
    def __init__(self, trial, input_dim):
        super().__init__()
        layers = []
        in_features = input_dim
        
        # Search Space
        n_layers = trial.suggest_int("n_layers", 2, 6)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        activation_name = trial.suggest_categorical("act", ["GELU", "SiLU", "Mish"]) 

        for i in range(n_layers):
            out_features = trial.suggest_int(f"units_l{i}", 128, 1024, step=64)
            
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features)) # Ổn định gradient
            
            if activation_name == "GELU":
                layers.append(nn.GELU())
            elif activation_name == "SiLU":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.Mish()) # Activation hiện đại
                
            layers.append(nn.Dropout(dropout))
            in_features = out_features
            
        # Output layer (Nằm ngoài vòng lặp for)
        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#%%
# --- [CELL 20]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 21}
# === BEFORE (original) ===
# def objective(trial):
#     # Tự động chọn device (Không cần queue)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 
#     # --- Hyperparams ---
#     lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
#     batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
#     optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RAdam"])
#     
#     # --- Data Setup (Load lên GPU 1 lần) ---
#     # X_train, y_train là biến toàn cục (global variables)
#     X_train_t = torch.tensor(X_train, device=device, dtype=torch.float32)
#     y_train_t = torch.tensor(y_train, device=device, dtype=torch.float32).unsqueeze(1)
#     X_val_t = torch.tensor(X_val, device=device, dtype=torch.float32)
#     # y_val giữ nguyên numpy để tính AUC bằng sklearn
#     
#     num_samples = X_train_t.size(0)
#     num_batches = int(np.ceil(num_samples / batch_size))
#     
#     # --- Model Init ---
#     input_dim = X_train.shape[1] # Lấy số features động
#     model = AdvancedFraudANN(trial, input_dim).to(device)
#     
#     # Loss function
#     pos_weight = torch.tensor(pos_weight_val, device=device)
#     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#     
#     if optimizer_name == "AdamW":
#         optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
#     else:
#         optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-5)
#         
#     # --- Mixed Precision ---
#     scaler_amp = GradScaler()
# 
#     # --- Training Loop ---
#     EPOCHS = 2 #20 # Số epoch vừa đủ để tune hyperparameters # ---------change for reproducing purposes----------
#     indices = torch.arange(num_samples, device=device)
#     
#     for epoch in range(EPOCHS):
#         model.train()
#         indices = indices[torch.randperm(num_samples, device=device)]
#         
#         for i in range(num_batches):
#             start = i * batch_size
#             end = min(start + batch_size, num_samples)
#             idx = indices[start:end]
# 
#             X_batch = X_train_t[idx]
#             y_batch = y_train_t[idx]
#             
#             optimizer.zero_grad()
#             
#             with autocast():
#                 outputs = model(X_batch)
#                 loss = criterion(outputs, y_batch)
#             
#             scaler_amp.scale(loss).backward()
#             scaler_amp.step(optimizer)
#             scaler_amp.update()
#         
#         # --- Validation ---
#         model.eval()
#         with torch.no_grad():
#             with autocast():
#                 val_logits = model(X_val_t)
#                 val_probs = torch.sigmoid(val_logits)
#             
#             # Chuyển về CPU để tính toán với Sklearn
#             val_preds = val_probs.detach().cpu().numpy()
#             val_auc = roc_auc_score(y_val, val_preds)
#         
#         # --- Optuna Pruning ---
#         trial.report(val_auc, epoch)
#         if trial.should_prune():
#             raise optuna.exceptions.TrialPruned()
# 
#     return val_auc

# === AFTER (edited) ===
def objective(trial):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RAdam"])



    X_train_t = torch.tensor(X_train, device=device, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, device=device, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, device=device, dtype=torch.float32)


    num_samples = X_train_t.size(0)
    num_batches = int(np.ceil(num_samples / batch_size))


    input_dim = X_train.shape[1]
    model = AdvancedFraudANN(trial, input_dim).to(device)


    pos_weight = torch.tensor(pos_weight_val, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        optimizer = optim.RAdam(model.parameters(), lr=lr, weight_decay=1e-5)


    scaler_amp = GradScaler()


    EPOCHS = 2
    indices = torch.arange(num_samples, device=device)

    for epoch in range(EPOCHS):
        model.train()
        indices = indices[torch.randperm(num_samples, device=device)]

        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            idx = indices[start:end]

            X_batch = X_train_t[idx]
            y_batch = y_train_t[idx]

            optimizer.zero_grad()

            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()


        model.eval()
        with torch.no_grad():
            with autocast():
                val_logits = model(X_val_t)
                val_probs = torch.sigmoid(val_logits)


            val_preds = val_probs.detach().cpu().numpy()
            
            # Filter out NaN values before computing AUC
            mask = ~np.isnan(val_preds)
            val_preds_clean = val_preds[mask]
            y_val_clean = y_val[mask]
            
            # Only compute AUC if we have valid samples
            if len(val_preds_clean) > 0 and len(y_val_clean) > 0:
                val_auc = roc_auc_score(y_val_clean, val_preds_clean)
            else:
                val_auc = 0.0


        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_auc

#%%
# --- [CELL 21]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 22}
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=3) # Chạy tuần tự, không cần n_jobs > 1 nếu dùng GPU # 100
    
    print("Best params:", study.best_params)
    print("Best AUC:", study.best_value)