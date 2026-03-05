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
# Create sample train and test data
np.random.seed(42)

# Sample size
train_size = 1000
test_size = 500

# Feature data
data = {
    "GarageArea": np.random.uniform(0, 1000, train_size + test_size),
    "LotArea": np.random.uniform(2000, 100000, train_size + test_size),
    "LotFrontage": np.random.uniform(40, 150, train_size + test_size),
    "OverallQual": np.random.randint(1, 10, train_size + test_size),
    "PoolArea": np.random.uniform(0, 800, train_size + test_size),
    "MSSubClass": np.random.randint(20, 200, train_size + test_size),
    "YearBuilt": np.random.randint(1950, 2020, train_size + test_size),
    "GrLivArea": np.random.uniform(500, 4000, train_size + test_size),
    "BedroomAbvGr": np.random.randint(0, 8, train_size + test_size),
    "LowQualFinSF": np.random.uniform(0, 500, train_size + test_size),
    "TotRmsAbvGrd": np.random.randint(2, 14, train_size + test_size),
    "Id": np.arange(train_size + test_size),
}

df = pd.DataFrame(data)

# Add SalePrice target (only for training data)
df["SalePrice"] = np.random.uniform(50000, 800000, train_size + test_size)

# Split into train and test
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()
test["SalePrice"] = np.nan  # No target for test data

# Concat as original code does
df = pd.concat([train, test], ignore_index=True)

# Select columns
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

# Fixed barplot - using keyword arguments for modern seaborn
for col in num_cols:
    sns.barplot(x=col, y=df[col], data=df)
    plt.title(col)
    plt.show(block=True)

for col in num_cols:
    sns.boxplot(data=df, x=col)
    plt.show(block=True)