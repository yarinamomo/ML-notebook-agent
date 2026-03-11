# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
app_train = pd.read_csv('data/application_train.csv.zip')
app_test=pd.read_csv('data/application_test.csv.zip')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
app_train.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan}, inplace = True)
app_test.replace({'XNA': np.nan, 'XNP': np.nan, 'Unknown': np.nan}, inplace = True)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
app_test.drop(app_train.columns[app_train.isnull().mean()>0.4],axis=1, inplace=True)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
app_train.drop(app_train.columns[app_train.isnull().mean()>0.4],axis=1, inplace=True)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
for i in Cat_columns_lower_percentage_nan:
    app_test[i].fillna(app_train[i].mode()[0], inplace=True)
    app_train[i].fillna(app_train[i].mode()[0], inplace=True)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
col_mod_transfrom = [i for i in num_columns_lower_percentage_nan if i not in ['EXT_SOURCE_2', 'AMT_ANNUITY','AMT_GOODS_PRICE']]
col_mean_transform = ['EXT_SOURCE_2', 'AMT_ANNUITY']

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
for i in col_mod_transfrom:
    app_test[i].fillna(app_train[i].mode()[0], inplace=True)
    app_train[i].fillna(app_train[i].mode()[0], inplace=True)
for i in col_mean_transform:
    app_test[i].fillna(app_train[i].mean(), inplace=True)
    app_train[i].fillna(app_train[i].mean(), inplace=True)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
app_train['AMT_GOODS_PRICE'].fillna(app_train['AMT_GOODS_PRICE'].median(),inplace = True)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
all_numerical_cols = list(app_train.select_dtypes(exclude='object').columns)

cont_cols = [col for col in all_numerical_cols if col != "TARGET" and col[:5]!='FLAG_']

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
app_train['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True) 
app_test['DAYS_EMPLOYED'].replace({365243:np.nan},inplace=True)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
app_train = app_train[app_train['AMT_INCOME_TOTAL'] != 117000000.0]

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
cat_col = app_train.select_dtypes('object')
cat_col.describe()

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].fillna(app_test['ORGANIZATION_TYPE'].mode()[0])

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 20}
app_test['EXT_SOURCE_3'] = app_test['EXT_SOURCE_3'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['EXT_SOURCE_3'].transform('mean'))
app_train['EXT_SOURCE_3'] = app_train['EXT_SOURCE_3'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['EXT_SOURCE_3'].transform('mean'))

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 21}
app_test['DAYS_EMPLOYED'] = app_test['DAYS_EMPLOYED'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].transform('mean'))
app_train['DAYS_EMPLOYED'] = app_train['DAYS_EMPLOYED'].fillna(app_train.groupby(['OCCUPATION_TYPE'])['DAYS_EMPLOYED'].transform('mean'))

#%%
# --- [CELL 21]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 22}
proper_days_empolyed_df = app_train
proper_days_empolyed_df['YEARS_EMPLOYED'] = proper_days_empolyed_df['DAYS_EMPLOYED']/-365.25

#%%
# --- [CELL 22]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 23}
app_test['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'},inplace=True)
app_train['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'},inplace=True)

app_test['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)
app_train['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'},inplace=True)

#%%
# --- [CELL 23]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 24}
others = app_train['ORGANIZATION_TYPE'].value_counts().index[15:]
label = 'Others'
app_train['ORGANIZATION_TYPE'] = app_train['ORGANIZATION_TYPE'].replace(others, label)
app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].replace(others, label)

#%%
# --- [CELL 24]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 25}
app_train.drop(['YEARS_EMPLOYED'], axis = 1,inplace=True)

#%%
# --- [CELL 25]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 26}
app_train = app_train.drop(columns=['CNT_FAM_MEMBERS','LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_WORK_REGION', 'OBS_60_CNT_SOCIAL_CIRCLE'])
app_test = app_test.drop(columns=['CNT_FAM_MEMBERS','LIVE_REGION_NOT_WORK_REGION', 'REG_REGION_NOT_WORK_REGION', 'OBS_60_CNT_SOCIAL_CIRCLE'])

#%%
# --- [CELL 26]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 27}
cols_to_remove = ['AMT_CREDIT', 'CNT_FAM_MEMBERS', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'OBS_60_CNT_SOCIAL_CIRCLE','SK_ID_CURR']
cont_cols = list(set(cont_cols) - set(cols_to_remove))
cont_cols

#%%
# --- [CELL 27]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 28}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 29}
for i in cont_cols:
    app_train[i] = impute_outliers_IQR(app_train[i])
    app_test[i] = impute_outliers_IQR(app_test[i])

#%%
# --- [CELL 29]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 30}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 31}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 32}
enc = TargetEncoder()
app_train[cat_col.columns] = enc.fit_transform(app_train[cat_col.columns], app_train['TARGET'])

#%%
# --- [CELL 32]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 33}
app_test[cat_col.columns] = enc.transform(app_test[cat_col.columns])

#%%
# --- [CELL 33]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 34}
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Выделение целевой переменной и признаков
y = app_train['TARGET']
X = app_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)

# Предобработка данных
X = pd.get_dummies(X, dummy_na=True)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Балансировка классов
over_sampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
under_sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
steps = [('o', over_sampler), ('u', under_sampler)]
pipeline = Pipeline(steps=steps)
X_train, y_train = pipeline.fit_resample(X_train, y_train)

# Создание и обучение модели логистической регрессии
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# Оценка качества модели
y_pred = lr.predict(X_test)
score = lr.score(X_test, y_test)
print(f"Accuracy: {score:.2f}")

#%%
# --- [CELL 34]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 35}
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
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 36}
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

# Get the number of columns from X_train (72 after encoding)
# Using the default value of 72 based on the error message
num_features = 72

pset = gp.PrimitiveSet("MAIN", arity=num_features)
pset.addPrimitive(np.add, arity=2)
pset.addPrimitive(np.subtract, arity=2)
pset.addPrimitive(np.multiply, arity=2)
pset.addPrimitive(np.maximum, arity=2)
pset.addPrimitive(np.minimum, arity=2)
pset.addPrimitive(protected_division, arity=2)
pset.addPrimitive(protected_sqrt, arity=1)
pset.addPrimitive(protected_log, arity=1)
pset.addPrimitive(np.sin, arity=1)
pset.addPrimitive(np.cos, arity=1)
pset.addTerminal(0)
pset.addTerminal(1)

#%%
# --- [CELL 36]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 37}
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
    # Generate new feature and stack with original features
    new_feature = np.array([expr(*row) for row in X])
    # Keep all original features and add the GP-generated feature
    return np.column_stack([X, new_feature])


def evaluate_fitness(individual):

    X_train_gp = transform_gp_structure(individual, X_train)
    rf_model_gp = lr
    rf_model_gp.fit(X_train_gp, y_train)
    X_test_gp = transform_gp_structure(individual, X_test)
    y_pred_gp = rf_model_gp.predict_proba(X_test_gp)[:, 1]
    auc_gp = roc_auc_score(y_test, y_pred_gp)
    return auc_gp,

#%%
# --- [CELL 37]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 38}
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
# cell_state: edited
# execution_status: {'status': 'timeout', 'done': True, 'execution_count': None}
# === BEFORE (original) ===
# import random # fix missing import and vars for crash isolation purposes
# crossover_prob = 0.5
# mutation_prob = 0.5
# 
# pop_size = 100  # Размер популяции
# num_generations = 50  # Количество поколений
# 
# # Создание начальной популяции
# population = toolbox.population(n=pop_size)
# 
# # Основной цикл эволюции
# for generation in range(num_generations):
#     # Оценка фитнеса
#     fitnesses = map(evaluate_fitness, population)
#     for individual, fitness in zip(population, fitnesses):
#         individual.fitness.values = fitness
# 
#     # Выбор следующего поколения
#     offspring = toolbox.select(population, len(population))
# 
#     # Клонирование выбранных индивидуумов
#     offspring = list(map(toolbox.clone, offspring))
# 
#     # Применение операторов скрещивания и мутации
#     for child1, child2 in zip(offspring[::2], offspring[1::2]):
#         if random.random() < crossover_prob:
#             toolbox.mate(child1, child2)
#             del child1.fitness.values
#             del child2.fitness.values
# 
#     for mutant in offspring:
#         if random.random() < mutation_prob:
#             toolbox.mutate(mutant)
#             del mutant.fitness.values
# 
#     # Замена старого поколения потомками
#     population[:] = offspring

# === AFTER (edited) ===
import random
crossover_prob = 0.5
mutation_prob = 0.5

pop_size = 100
num_generations = 50


population = toolbox.population(n=pop_size)


for generation in range(num_generations):

    fitnesses = map(evaluate_fitness, population)
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness


    offspring = toolbox.select(population, len(population))


    offspring = list(map(toolbox.clone, offspring))


    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < crossover_prob:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < mutation_prob:
            toolbox.mutate(mutant)
            del mutant.fitness.values


    population[:] = offspring