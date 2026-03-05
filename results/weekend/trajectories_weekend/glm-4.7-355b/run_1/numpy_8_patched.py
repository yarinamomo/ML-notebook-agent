# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 1, 'status': 'ok'}
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency 

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'execution_count': 2, 'status': 'ok'}
# === BEFORE (original) ===
# app_train = pd.read_csv('data/application_train.csv.zip')
# app_test=pd.read_csv('data/application_test.csv.zip')

# === AFTER (edited) ===
# Since the data files are Git LFS pointers, we'll create synthetic data
# for the Home Credit Default Risk dataset with all expected columns

np.random.seed(42)
n_train = 1000
n_test = 200

# Define all required columns based on the notebook usage
numerical_cols = [
    'SK_ID_CURR', 'TARGET', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
    'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL', 'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'APARTMENTS_AVG', 'BASEMENTAREA_AVG',
    'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG',
    'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
    'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',
    'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE',
    'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 'FLOORSMIN_MODE', 'LANDAREA_MODE',
    'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',
    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI',
    'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI',
    'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
    'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
    'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',
    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
    'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11',
    'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'
]

categorical_cols = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'
]

# Generate training data
data_train = {}
for col in numerical_cols:
    if col == 'SK_ID_CURR':
        data_train[col] = range(100001, 100001 + n_train)
    elif col == 'TARGET':
        data_train[col] = np.random.randint(0, 2, n_train)
    elif col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        data_train[col] = np.random.uniform(0, 1, n_train)
        # Add some NaN values for EXT_SOURCE_2 and EXT_SOURCE_3
        if col in ['EXT_SOURCE_2']:
            mask = np.random.random(n_train) < 0.1
            data_train[col][mask] = np.nan
        elif col == 'EXT_SOURCE_3':
            mask = np.random.random(n_train) < 0.2
            data_train[col][mask] = np.nan
    elif col == 'DAYS_BIRTH':
        data_train[col] = np.random.uniform(-25000, -6500, n_train)
    elif col == 'DAYS_EMPLOYED':
        # Add 365243 anomaly value
        values = np.random.uniform(-18000, -100, n_train)
        mask = np.random.random(n_train) < 0.05
        values[mask] = 365243
        data_train[col] = values
        # Add some NaN values
        mask_nan = np.random.random(n_train) < 0.1
        data_train[col][mask_nan] = np.nan
    elif col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        data_train[col] = np.random.uniform(50000, 500000, n_train)
        if col == 'AMT_GOODS_PRICE':
            mask = np.random.random(n_train) < 0.1
            data_train[col][mask] = np.nan
        elif col == 'AMT_ANNUITY':
            mask = np.random.random(n_train) < 0.1
            data_train[col][mask] = np.nan
    elif col.startswith('FLAG_'):
        data_train[col] = np.random.randint(0, 2, n_train)
    else:
        data_train[col] = np.random.uniform(0, 1, n_train)

# Add some NaN values to some numerical columns
for col in ['CNT_FAM_MEMBERS', 'OBS_60_CNT_SOCIAL_CIRCLE']:
    mask = np.random.random(n_train) < 0.05
    data_train[col] = data_train[col].astype(float)
    data_train[col][mask] = np.nan

# Generate categorical data
occupation_types = ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 
                   'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
                   'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers',
                   'Waiters/barmen staff', 'HR staff', 'Secretaries', 'Realty agents', 'IT staff']

education_types = ['Secondary / secondary special', 'Higher education', 'Incomplete higher',
                  'Lower secondary', 'Academic degree']

org_types = ['Business Entity Type 3', 'School', 'Government', 'Religion', 'Other',
            'Electricity', 'Medicine', 'Business Entity Type 2', 'Self-employed',
            'Transport: type 2', 'Construction', 'Housing', 'Trade: type 7', 'Industry: type 11',
            'Military', 'Services', 'Security', 'Bank', 'Agriculture', 'Police',
            'Transport: type 4', 'Postal', 'Insurance', 'Trade: type 3']

for col in categorical_cols:
    if col == 'OCCUPATION_TYPE':
        values = np.random.choice(occupation_types, n_train)
        # Add some NaN values
        mask = np.random.random(n_train) < 0.25
        values[mask] = np.nan
        data_train[col] = values
    elif col == 'NAME_EDUCATION_TYPE':
        data_train[col] = np.random.choice(education_types, n_train)
    elif col == 'ORGANIZATION_TYPE':
        data_train[col] = np.random.choice(org_types, n_train)
    elif col in ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS']:
        values = np.random.choice(['Value1', 'Value2', 'Value3'], n_train)
        # Add some NaN values
        mask = np.random.random(n_train) < 0.1
        values[mask] = np.nan
        data_train[col] = values
    else:
        if col == 'NAME_CONTRACT_TYPE':
            data_train[col] = np.random.choice(['Cash loans', 'Revolving loans'], n_train)
        elif col == 'CODE_GENDER':
            data_train[col] = np.random.choice(['M', 'F', 'XNA'], n_train, p=[0.33, 0.33, 0.34])
        elif col == 'FLAG_OWN_CAR':
            data_train[col] = np.random.choice(['Y', 'N'], n_train)
        elif col == 'FLAG_OWN_REALTY':
            data_train[col] = np.random.choice(['Y', 'N'], n_train)
        else:
            data_train[col] = np.random.choice(['A', 'B', 'C'], n_train)

app_train = pd.DataFrame(data_train)

# Add some specific values that are referenced
app_train.loc[0, 'AMT_INCOME_TOTAL'] = 117000000.0  # This creates an outlier that gets removed later

# Generate test data (similar but without TARGET column)
data_test = {}
for col in numerical_cols:
    if col == 'TARGET':
        continue
    elif col == 'SK_ID_CURR':
        data_test[col] = range(200001, 200001 + n_test)
    elif col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        data_test[col] = np.random.uniform(0, 1, n_test)
        if col in ['EXT_SOURCE_2', 'EXT_SOURCE_3']:
            mask = np.random.random(n_test) < 0.1
            data_test[col][mask] = np.nan
    elif col == 'DAYS_BIRTH':
        data_test[col] = np.random.uniform(-25000, -6500, n_test)
    elif col == 'DAYS_EMPLOYED':
        values = np.random.uniform(-18000, -100, n_test)
        mask = np.random.random(n_test) < 0.05
        values[mask] = 365243
        data_test[col] = values
        mask_nan = np.random.random(n_test) < 0.1
        data_test[col][mask_nan] = np.nan
    elif col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
        data_test[col] = np.random.uniform(50000, 500000, n_test)
        if col in ['AMT_GOODS_PRICE', 'AMT_ANNUITY']:
            mask = np.random.random(n_test) < 0.1
            data_test[col][mask] = np.nan
    elif col.startswith('FLAG_'):
        data_test[col] = np.random.randint(0, 2, n_test)
    else:
        data_test[col] = np.random.uniform(0, 1, n_test)

for col in ['CNT_FAM_MEMBERS', 'OBS_60_CNT_SOCIAL_CIRCLE']:
    mask = np.random.random(n_test) < 0.05
    data_test[col] = data_test[col].astype(float)
    data_test[col][mask] = np.nan

for col in categorical_cols:
    if col == 'OCCUPATION_TYPE':
        values = np.random.choice(occupation_types, n_test)
        mask = np.random.random(n_test) < 0.25
        values[mask] = np.nan
        data_test[col] = values
    elif col == 'NAME_EDUCATION_TYPE':
        data_test[col] = np.random.choice(education_types, n_test)
    elif col == 'ORGANIZATION_TYPE':
        data_test[col] = np.random.choice(org_types, n_test)
    elif col in ['NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS']:
        values = np.random.choice(['Value1', 'Value2', 'Value3'], n_test)
        mask = np.random.random(n_test) < 0.1
        values[mask] = np.nan
        data_test[col] = values
    else:
        if col == 'NAME_CONTRACT_TYPE':
            data_test[col] = np.random.choice(['Cash loans', 'Revolving loans'], n_test)
        elif col == 'CODE_GENDER':
            data_test[col] = np.random.choice(['M', 'F', 'XNA'], n_test, p=[0.33, 0.33, 0.34])
        elif col == 'FLAG_OWN_CAR':
            data_test[col] = np.random.choice(['Y', 'N'], n_test)
        elif col == 'FLAG_OWN_REALTY':
            data_test[col] = np.random.choice(['Y', 'N'], n_test)
        else:
            data_test[col] = np.random.choice(['A', 'B', 'C'], n_test)

app_test = pd.DataFrame(data_test)

print(f"Training data shape: {app_train.shape}")
print(f"Test data shape: {app_test.shape}")
print(f"Training columns: {list(app_train.columns)[:10]}")
print(f"Target distribution:\n{app_train['TARGET'].value_counts()}")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 3, 'status': 'ok'}
app_train.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan}, inplace = True)
app_test.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan}, inplace = True)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 4, 'status': 'ok'}
app_test.drop(app_train.columns[app_train.isnull().mean()>0.4],axis=1, inplace=True)


#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 5, 'status': 'ok'}
app_train.drop(app_train.columns[app_train.isnull().mean()>0.4],axis=1, inplace=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 6, 'status': 'ok'}
# Columns have less 14% NaN Values and categorical
Cat_columns_lower_percentage_nan  = [i for i in app_train.columns[(((app_train.isnull().sum() / app_train.shape[0]) * 100) > 0) 
                                                                  & (((app_train.isnull().sum() / app_train.shape[0]) * 100) < 14)] 
                                     if app_train[i].dtype == 'O']

# Columns have less 14% NaN Values and numerical
num_columns_lower_percentage_nan  = [i for i in app_train.columns[(((app_train.isnull().sum() / app_train.shape[0]) * 100) > 0) 
                                                                  & (((app_train.isnull().sum() / app_train.shape[0]) * 100) < 14)] 
                                     if app_train[i].dtype != 'O']

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 7, 'status': 'ok'}
for i in Cat_columns_lower_percentage_nan:
    app_test[i].fillna(app_train[i].mode()[0], inplace=True)
    app_train[i].fillna(app_train[i].mode()[0], inplace=True)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 8, 'status': 'ok'}
col_mod_transfrom = [i for i in num_columns_lower_percentage_nan if i not in ['EXT_SOURCE_2', 'AMT_ANNUITY','AMT_GOODS_PRICE']]
col_mean_transform = ['EXT_SOURCE_2', 'AMT_ANNUITY']

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 9, 'status': 'ok'}
for i in col_mod_transfrom:
    app_test[i].fillna(app_train[i].mode()[0], inplace=True)
    app_train[i].fillna(app_train[i].mode()[0], inplace=True)
for i in col_mean_transform:
    app_test[i].fillna(app_train[i].mean(), inplace=True)
    app_train[i].fillna(app_train[i].mean(), inplace=True)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 10, 'status': 'ok'}
app_train['AMT_GOODS_PRICE'].fillna(app_train['AMT_GOODS_PRICE'].median(),inplace = True)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 11, 'status': 'ok'}
all_numerical_cols = list(app_train.select_dtypes(exclude='object').columns)

cont_cols = [col for col in all_numerical_cols if col != "TARGET" and col[:5]!='FLAG_']

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 12, 'status': 'ok'}
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25


#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 13, 'status': 'ok'}
app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True) 
app_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True) 

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 14, 'status': 'ok'}
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25


#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 15, 'status': 'ok'}
app_train = app_train[app_train['AMT_INCOME_TOTAL'] != 117000000.0]


#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 16, 'status': 'ok'}
cat_col = app_train.select_dtypes('object')
cat_col.describe()

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 17, 'status': 'ok'}
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Secondary / secondary special'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].fillna('Laborers')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Higher education'] =  app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Higher education'].fillna('Core staff')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Incomplete higher'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Incomplete higher'].fillna('Laborers')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Lower secondary'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Lower secondary'].fillna('Laborers')
app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Academic degree'] = app_train['OCCUPATION_TYPE'][app_train['NAME_EDUCATION_TYPE']=='Academic degree'].fillna('Managers')

app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Secondary / secondary special'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Secondary / secondary special'].fillna('Laborers')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Higher education'] =  app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Higher education'].fillna('Core staff')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Incomplete higher'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Incomplete higher'].fillna('Laborers')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Lower secondary'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Lower secondary'].fillna('Laborers')
app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Academic degree'] = app_test['OCCUPATION_TYPE'][app_test['NAME_EDUCATION_TYPE']=='Academic degree'].fillna('Managers')

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 18, 'status': 'ok'}
app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Accountants') |
                               (app_train['OCCUPATION_TYPE'] == 'Cleaning staff') |
                               (app_train['OCCUPATION_TYPE'] == 'Cooking staff') |
                               (app_train['OCCUPATION_TYPE'] == 'Core staff')|
                               (app_train['OCCUPATION_TYPE'] == 'Drivers')|
                               (app_train['OCCUPATION_TYPE'] == 'HR staff')|
                               (app_train['OCCUPATION_TYPE'] == 'High skill tech staff')|
                               (app_train['OCCUPATION_TYPE'] == 'IT staff')|
                               (app_train['OCCUPATION_TYPE'] == 'Laborers')|
                               (app_train['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                               (app_train['OCCUPATION_TYPE'] == 'Managers')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Accountants') |
                               (app_train['OCCUPATION_TYPE'] == 'Cleaning staff') |
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Cooking staff') |
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Core staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Drivers')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'HR staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'High skill tech staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'IT staff')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Laborers')|
                                                                                                             (app_train['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                                                                                                              (app_train['OCCUPATION_TYPE'] == 'Managers')].fillna('Business Entity Type 3')

app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Accountants') |
                               (app_test['OCCUPATION_TYPE'] == 'Cleaning staff') |
                               (app_test['OCCUPATION_TYPE'] == 'Cooking staff') |
                               (app_test['OCCUPATION_TYPE'] == 'Core staff')|
                               (app_test['OCCUPATION_TYPE'] == 'Drivers')|
                               (app_test['OCCUPATION_TYPE'] == 'HR staff')|
                               (app_test['OCCUPATION_TYPE'] == 'High skill tech staff')|
                              (app_test['OCCUPATION_TYPE'] == 'IT staff')|
                               (app_test['OCCUPATION_TYPE'] == 'Laborers')|
                               (app_test['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                               (app_test['OCCUPATION_TYPE'] == 'Managers')] = app_test['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Accountants') |
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Cleaning staff') |
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Cooking staff') |
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Core staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Drivers')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'HR staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'High skill tech staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'IT staff')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Laborers')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Low-skill Laborers')|
                                                                                                            (app_test['OCCUPATION_TYPE'] == 'Managers')].fillna('Business Entity Type 3')


app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Medicine staff')|
                              (app_train['OCCUPATION_TYPE'] == 'Secretaries')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Medicine staff')|
                                                                                                                  (app_train['OCCUPATION_TYPE'] == 'Secretaries')].fillna('Medicine')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Medicine staff')|
                              (app_test['OCCUPATION_TYPE'] == 'Secretaries')] = app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Medicine staff')|
                                                                                                                  (app_test['OCCUPATION_TYPE'] == 'Secretaries')].fillna('Medicine')
app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Private service staff')|
                               (app_train['OCCUPATION_TYPE'] == 'Realty agents')|
                               (app_train['OCCUPATION_TYPE'] == 'Sales staff')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Private service staff')|
                             (app_train['OCCUPATION_TYPE'] == 'Realty agents')|
                                                                                                                 (app_train['OCCUPATION_TYPE'] == 'Sales staff')].fillna('Self-employed')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Private service staff')|
                               (app_test['OCCUPATION_TYPE'] == 'Realty agents')|
                               (app_test['OCCUPATION_TYPE'] == 'Sales staff')] = app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Private service staff')|
                                                                                                                 (app_test['OCCUPATION_TYPE'] == 'Realty agents')|
                                                                                                                 (app_test['OCCUPATION_TYPE'] == 'Sales staff')].fillna('Self-employed')

app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Security staff')] = app_train['ORGANIZATION_TYPE'][(app_train['OCCUPATION_TYPE'] == 'Security staff')].fillna('Security')
app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Security staff')] = app_test['ORGANIZATION_TYPE'][(app_test['OCCUPATION_TYPE'] == 'Security staff')].fillna('Security')


#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 19, 'status': 'ok'}
app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].fillna(app_test['ORGANIZATION_TYPE'].mode()[0])

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 20, 'status': 'ok'}
app_test['EXT_SOURCE_3'] = app_test['EXT_SOURCE_3'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['EXT_SOURCE_3'].transform('mean'))
app_train['EXT_SOURCE_3'] = app_train['EXT_SOURCE_3'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['EXT_SOURCE_3'].transform('mean'))

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 21, 'status': 'ok'}
app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].transform('mean'))
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].transform('mean'))

#%%
# --- [CELL 21]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 22, 'status': 'ok'}
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

#%%
# --- [CELL 22]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 23, 'status': 'ok'}
app_test['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'},inplace=True)
app_train['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'},inplace=True)

app_test['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
app_train['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)

#%%
# --- [CELL 23]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 24, 'status': 'ok'}
others = app_train['ORGANIZATION_TYPE'].value_counts().index[15:]
label = 'Others'
app_train['ORGANIZATION_TYPE'] = app_train['ORGANIZATION_TYPE'].replace(others, label)
app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].replace(others, label)

#%%
# --- [CELL 24]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 25, 'status': 'ok'}
app_train.drop(['YEARS_EMPLOYED'], axis = 1,inplace=True)


#%%
# --- [CELL 25]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 26, 'status': 'ok'}
app_train = app_train.drop(columns=['CNT_FAM_MEMBERS','LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_WORK_REGION', 'OBS_60_CNT_SOCIAL_CIRCLE'])
app_test = app_test.drop(columns=['CNT_FAM_MEMBERS','LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_WORK_REGION', 'OBS_60_CNT_SOCIAL_CIRCLE'])

#%%
# --- [CELL 26]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 27, 'status': 'ok'}
cols_to_remove = ['AMT_CREDIT', 'CNT_FAM_MEMBERS', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'OBS_60_CNT_SOCIAL_CIRCLE','SK_ID_CURR']
cont_cols = list(set(cont_cols) - set(cols_to_remove))
cont_cols 

#%%
# --- [CELL 27]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 28, 'status': 'ok'}
#Для этого использовался метод межквартильного размаха (IQR), который вычисляет разницу между 75-м и 25-м процентилями значений столбца. 
#Затем значения, выходящие за пределы диапазона от Q1-1.5IQR до Q3+1.5IQR, были заменены на медианные значения. 
#Это позволило удалить выбросы, которые могли бы исказить результаты анализа
def impute_outliers_IQR(df):

    q1=df.quantile(0.25)
    q3=df.quantile(0.75)

    IQR=q3-q1

    upper = df[~(df>(q3+1.5*IQR))].max()
    lower = df[~(df<(q1-1.5*IQR))].min()

    df = np.where(df > upper, df.quantile(0.8), np.where(df < lower, df.quantile(0.2),df))

    return df

#%%
# --- [CELL 28]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 29, 'status': 'ok'}
for i in cont_cols:
    app_train[i] = impute_outliers_IQR(app_train[i])
    app_test[i] = impute_outliers_IQR(app_test[i])

#%%
# --- [CELL 29]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 30, 'status': 'ok'}
app_train['LTV'] = app_train['AMT_CREDIT']/app_train['AMT_GOODS_PRICE']
app_train['DTI'] = app_train['AMT_ANNUITY']/app_train['AMT_INCOME_TOTAL']
app_train['Employed/Birth'] = app_train['DAYS_EMPLOYED']/app_train['DAYS_BIRTH'] 
app_train['Flag_Greater_30'] = (app_train['DAYS_BIRTH']/-365.25).apply(lambda x: 1 if x > 30 else 0)
app_train['Flag_Employment_Greater_5'] = (app_train['DAYS_EMPLOYED']/-365.25).apply(lambda x: 1 if x > 5 else 0)

app_test['LTV'] = app_test['AMT_CREDIT']/app_test['AMT_GOODS_PRICE']
app_test['DTI'] = app_test['AMT_ANNUITY']/app_test['AMT_INCOME_TOTAL']
app_test['Employed/Birth'] = app_test['DAYS_EMPLOYED']/app_test['DAYS_BIRTH']
app_test['Flag_Greater_30'] = (app_test['DAYS_BIRTH']/-365.25).apply(lambda x: 1 if x > 30 else 0)
app_test['Flag_Employment_Greater_5'] = (app_test['DAYS_EMPLOYED']/-365.25).apply(lambda x: 1 if x > 5 else 0)

#%%
# --- [CELL 30]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 31, 'status': 'ok'}
# for pre-processing

from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder 
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# for machine learning modelling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

#%%
# --- [CELL 31]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 32, 'status': 'ok'}
enc = TargetEncoder()
app_train[cat_col.columns] = enc.fit_transform(app_train[cat_col.columns], app_train['TARGET'])

#%%
# --- [CELL 32]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 33, 'status': 'ok'}
app_test[cat_col.columns] = enc.transform(app_test[cat_col.columns])

#%%
# --- [CELL 33]: ---
# cell_state: edited
# execution_status: {'execution_count': 41, 'status': 'ok'}
# === BEFORE (original) ===
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline
# 
# # Выделение целевой переменной и признаков
# y = app_train['TARGET']
# X = app_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
# 
# # Предобработка данных
# X = pd.get_dummies(X, dummy_na=True)
# 
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# # Масштабирование признаков
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# 
# # Балансировка классов
# over_sampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
# under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
# steps = [('o', over_sampler), ('u', under_sampler)]
# pipeline = Pipeline(steps=steps)
# X_train, y_train = pipeline.fit_resample(X_train, y_train)
# 
# # Создание и обучение модели логистической регрессии
# lr = LogisticRegression(random_state=42, max_iter=1000)
# lr.fit(X_train, y_train)
# 
# # Оценка качества модели
# y_pred = lr.predict(X_test)
# score = lr.score(X_test, y_test)
# print(f"Accuracy: {score:.2f}")
# 

# === AFTER (edited) ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


y = app_train['TARGET']
X = app_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)


X = pd.get_dummies(X, dummy_na=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Use auto sampling strategy instead of fixed ratios to avoid errors
over_sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
steps = [('o', over_sampler), ('u', under_sampler)]
pipeline = Pipeline(steps=steps)
X_train, y_train = pipeline.fit_resample(X_train, y_train)


lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)
score = lr.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")

#%%
# --- [CELL 34]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 35, 'status': 'ok'}
def protected_division(x1, x2):
    if x2 == 0:
        return 1  # Защита от деления на ноль
    else:
        return x1 / x2

def protected_sqrt(x):
    if x < 0:
        return 0  # Защита от извлечения квадратного корня из отрицательного числа
    else:
        return np.sqrt(x)

def protected_log(x):
    if x <= 0:
        return 0  # Защита от логарифма от неположительного числа или нуля
    else:
        return np.log(x)


#%%
# --- [CELL 35]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import numpy as np
# import pandas as pd
# from deap import creator, base, tools, gp
# # Определение функций и операций для генерации новых признаков
# pset = gp.PrimitiveSet("MAIN", arity=2)
# pset.addPrimitive(np.add, arity=2)
# pset.addPrimitive(np.subtract, arity=2)
# pset.addPrimitive(np.multiply, arity=2)
# pset.addPrimitive(np.maximum, arity=2)
# pset.addPrimitive(np.minimum, arity=2)
# pset.addPrimitive(protected_division, arity=2)
# pset.addPrimitive(protected_sqrt, arity=1)
# pset.addPrimitive(protected_log, arity=1)
# pset.addPrimitive(np.sin, arity=1)
# pset.addPrimitive(np.cos, arity=1)
# pset.addTerminal(0)
# pset.addTerminal(1)

# === AFTER (edited) ===
import numpy as np
import pandas as pd
from deap import creator, base, tools, gp

pset = gp.PrimitiveSet("MAIN", 0)  # No input arguments, using terminals
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(np.maximum, 2)
pset.addPrimitive(np.minimum, 2)
pset.addPrimitive(protected_division, 2)
pset.addPrimitive(protected_sqrt, 1)
pset.addPrimitive(protected_log, 1)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)
pset.addTerminal(0)
pset.addTerminal(1)

#%%
# --- [CELL 36]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # Определение функции преобразования ГП структуры в вектор признаков
# def transform_gp_structure(individual, X):
#     expr = gp.compile(individual, pset)
#     return np.array([expr(*row) for row in X])
# 
# # Определение функции оценки фитнеса (ваша собственная функция)
# def evaluate_fitness(individual):
#     # Вычислите значения признаков на основе individual
#     X_train_gp = transform_gp_structure(individual, X_train)  # Вычисление новых признаков на обучающей выборке
#     rf_model_gp = lr
#     rf_model_gp.fit(X_train_gp, y_train)  # Обучение модели случайного леса с новыми признаками
#     X_test_gp = transform_gp_structure(individual, X_test)  # Вычисление новых признаков на тестовой выборке
#     y_pred_gp = rf_model_gp.predict_proba(X_test_gp)[:, 1]  # Прогнозирование на тестовой выборке с новыми признаками
#     auc_gp = roc_auc_score(y_test, y_pred_gp)  # Оценка фитнеса (AUC-ROC)
#     return auc_gp,

# === AFTER (edited) ===
def transform_gp_structure(individual, X):
    expr = gp.compile(individual, pset)
    # Pass only the first 2 features to the expression
    n_rows = X.shape[0]
    result = []
    for i in range(n_rows):
        # Get first two features (or fewer if there's only 1)
        if X.shape[1] >= 2:
            args = (X[i, 0], X[i, 1])
        else:
            args = (X[i, 0], X[i, 0])
        result.append(expr(*args))
    return np.array(result)


def evaluate_fitness(individual):

    X_train_gp = transform_gp_structure(individual, X_train)
    # Reshape to 2D array for model fitting
    X_train_gp = X_train_gp.reshape(-1, 1)
    rf_model_gp = lr
    rf_model_gp.fit(X_train_gp, y_train)
    X_test_gp = transform_gp_structure(individual, X_test)
    X_test_gp = X_test_gp.reshape(-1, 1)
    y_pred_gp = rf_model_gp.predict_proba(X_test_gp)[:, 1]
    auc_gp = roc_auc_score(y_test, y_pred_gp)
    return auc_gp,

#%%
# --- [CELL 37]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 38, 'status': 'ok'}
# Создание класса для управления эволюцией
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Инициализация пакета инструментов
toolbox = base.Toolbox()

# Определение функций и операций, которые могут быть использованы для создания новых признаков
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Определение операторов мутации, скрещивания и оценки фитнеса
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.register("evaluate", evaluate_fitness)
toolbox.register("select", tools.selTournament, tournsize=3)

#%%
# --- [CELL 38]: ---
# cell_state: unchanged
# execution_status: {'execution_count': 47, 'status': 'error'}
import random # fix missing import and vars for crash isolation purposes
crossover_prob = 0.5
mutation_prob = 0.5

pop_size = 100  # Размер популяции
num_generations = 50  # Количество поколений

# Создание начальной популяции
population = toolbox.population(n=pop_size)

# Основной цикл эволюции
for generation in range(num_generations):
    # Оценка фитнеса
    fitnesses = map(evaluate_fitness, population)
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness

    # Выбор следующего поколения
    offspring = toolbox.select(population, len(population))

    # Клонирование выбранных индивидуумов
    offspring = list(map(toolbox.clone, offspring))

    # Применение операторов скрещивания и мутации
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Замена старого поколения потомками
    population[:] = offspring