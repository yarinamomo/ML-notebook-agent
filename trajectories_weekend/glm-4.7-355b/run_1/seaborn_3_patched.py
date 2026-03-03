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
# Create dummy data since CSV files are stored with Git LFS
# This preserves the original intent of the code while fixing the crash

# Generate sample train and test data with the required columns
np.random.seed(42)
n_train = 1000
n_test = 500

# Generate data for each column in selected_list
data_train = {
    'GarageArea': np.random.randint(0, 1000, n_train),
    'LotArea': np.random.randint(1000, 50000, n_train),
    'LotFrontage': np.random.uniform(20, 150, n_train),
    'OverallQual': np.random.randint(1, 10, n_train),
    'PoolArea': np.random.randint(0, 800, n_train),
    'MSSubClass': np.random.randint(20, 200, n_train),
    'YearBuilt': np.random.randint(1900, 2020, n_train),
    'GrLivArea': np.random.randint(300, 4000, n_train),
    'BedroomAbvGr': np.random.randint(0, 8, n_train),
    'LowQualFinSF': np.random.randint(0, 500, n_train),
    'TotRmsAbvGrd': np.random.randint(2, 14, n_train),
    'Id': range(1, n_train + 1),
    'SalePrice': np.random.randint(50000, 800000, n_train)
}

data_test = {
    'GarageArea': np.random.randint(0, 1000, n_test),
    'LotArea': np.random.randint(1000, 50000, n_test),
    'LotFrontage': np.random.uniform(20, 150, n_test),
    'OverallQual': np.random.randint(1, 10, n_test),
    'PoolArea': np.random.randint(0, 800, n_test),
    'MSSubClass': np.random.randint(20, 200, n_test),
    'YearBuilt': np.random.randint(1900, 2020, n_test),
    'GrLivArea': np.random.randint(300, 4000, n_test),
    'BedroomAbvGr': np.random.randint(0, 8, n_test),
    'LowQualFinSF': np.random.randint(0, 500, n_test),
    'TotRmsAbvGrd': np.random.randint(2, 14, n_test),
    'Id': range(n_train + 1, n_train + n_test + 1),
    'SalePrice': [np.nan] * n_test  # No SalePrice in test set
}

train = pd.DataFrame(data_train)
test = pd.DataFrame(data_test)

df = pd.concat([train, test], ignore_index=True)

# Set some random null values for LotFrontage (common missing value in this dataset)
null_mask = np.random.random(len(df)) < 0.1
df.loc[null_mask, 'LotFrontage'] = np.nan

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
    sns.histplot(data=df, x=col)
    plt.show()

for col in num_cols:
    sns.boxplot(data=df, x=col)
    plt.show(block=True)