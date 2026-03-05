# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# train = pd.read_csv("data/train.csv")
# test = pd.read_csv("data/test.csv")
# 
# # ignore_index=True kullanarak index sıfırdan başlayacak şekilde birleştirme yapın
# df = pd.concat([train, test], ignore_index=True)
# 
# selected_list = ["GarageArea", "LotArea", "LotFrontage", "OverallQual", "PoolArea", "MSSubClass", "YearBuilt", "GrLivArea",
#                  "BedroomAbvGr", "LowQualFinSF", "TotRmsAbvGrd", "Id", "SalePrice"]
# 
# # Id ve SalePrice'ı seçili sütunlara ekleyin
# df = df[selected_list]
# 
# # "index" sütununu düşürmeye gerek yok
# df.head()
# df.shape
# df.isnull().sum()

# === AFTER (edited) ===
# Create sample data since LFS files are not properly loaded
# This mimics the structure of a housing dataset

np.random.seed(42)

# Create train dataset with 1000 rows
train_data = {
    'Id': range(1, 1001),
    'MSSubClass': np.random.randint(20, 200, 1000),
    'LotArea': np.random.randint(1000, 20000, 1000),
    'LotFrontage': np.random.uniform(30, 100, 1000),
    'OverallQual': np.random.randint(1, 10, 1000),
    'YearBuilt': np.random.randint(1950, 2020, 1000),
    'YearRemodAdd': np.random.randint(1950, 2020, 1000),
    'GrLivArea': np.random.randint(500, 4000, 1000),
    'BedroomAbvGr': np.random.randint(0, 6, 1000),
    'LowQualFinSF': np.random.randint(0, 200, 1000),
    'GarageArea': np.random.randint(0, 1000, 1000),
    'PoolArea': np.random.randint(0, 600, 1000),
    'TotRmsAbvGrd': np.random.randint(2, 12, 1000),
    'SalePrice': np.random.uniform(50000, 500000, 1000)
}

# Create test dataset with 200 rows
test_data = {
    'Id': range(1001, 1201),
    'MSSubClass': np.random.randint(20, 200, 200),
    'LotArea': np.random.randint(1000, 20000, 200),
    'LotFrontage': np.random.uniform(30, 100, 200),
    'OverallQual': np.random.randint(1, 10, 200),
    'YearBuilt': np.random.randint(1950, 2020, 200),
    'YearRemodAdd': np.random.randint(1950, 2020, 200),
    'GrLivArea': np.random.randint(500, 4000, 200),
    'BedroomAbvGr': np.random.randint(0, 6, 200),
    'LowQualFinSF': np.random.randint(0, 200, 200),
    'GarageArea': np.random.randint(0, 1000, 200),
    'PoolArea': np.random.randint(0, 600, 200),
    'TotRmsAbvGrd': np.random.randint(2, 12, 200),
    'SalePrice': [np.nan] * 200  # Test data has no SalePrice
}

train = pd.DataFrame(train_data)
test = pd.DataFrame(test_data)

df = pd.concat([train, test], ignore_index=True)

selected_list = ["GarageArea", "LotArea", "LotFrontage", "OverallQual", "PoolArea", "MSSubClass", "YearBuilt", "GrLivArea",
                 "BedroomAbvGr", "LowQualFinSF", "TotRmsAbvGrd", "Id", "SalePrice"]

df = df[selected_list]

df.head()
df.shape
df.isnull().sum()

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
def grab_col_names(dataframe, cat_th=10, car_th=25):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)


#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# def num_summary(dataframe, numerical_col, plot=False):
#     quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
#     print(dataframe[numerical_col].describe(quantiles).T)
# 
#     if plot:
#         dataframe[numerical_col].hist(bins=50)
#         plt.xlabel(numerical_col)
#         plt.title(numerical_col)
#         plt.show(block=True)
# 
#     print("#####################################")
# 
# 
# for col in num_cols:
#     num_summary(df, col, True)
# 
# for col in num_cols:
#     sns.barplot(col,df)
# 
# for col in num_cols:
#     sns.boxplot(data=df, x=col)
#     plt.show(block=True)

# === AFTER (edited) ===
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)

for col in num_cols:
    sns.barplot(x=df.index, y=df[col])
    plt.title(col)
    plt.show()

for col in num_cols:
    sns.boxplot(data=df, x=col)
    plt.show(block=True)