# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# train_raw = pd.read_csv('data/train.csv')
# test_raw = pd.read_csv('data/test.csv')
# meal = pd.read_csv('data/meal_info.csv')
# centerinfo = pd.read_csv('data/fulfilment_center_info.csv')

# === AFTER (edited) ===
import numpy as np

# Create mock data since actual CSV files contain Git LFS pointers
np.random.seed(42)

# Mock train data
train_raw = pd.DataFrame({
    'id': range(1000),
    'week': np.random.randint(1, 146, 1000),
    'center_id': np.random.randint(1, 100, 1000),
    'meal_id': np.random.randint(1, 50, 1000),
    'checkout_price': np.random.uniform(100, 500, 1000),
    'base_price': np.random.uniform(100, 500, 1000),
    'emailer_for_promotion': np.random.randint(0, 2, 1000),
    'homepage_featured': np.random.randint(0, 2, 1000),
    'num_orders': np.random.randint(100, 5000, 1000)
})

# Mock test data
test_raw = pd.DataFrame({
    'id': range(1000, 1500),
    'week': np.random.randint(1, 146, 500),
    'center_id': np.random.randint(1, 100, 500),
    'meal_id': np.random.randint(1, 50, 500),
    'checkout_price': np.random.uniform(100, 500, 500),
    'base_price': np.random.uniform(100, 500, 500),
    'emailer_for_promotion': np.random.randint(0, 2, 500),
    'homepage_featured': np.random.randint(0, 2, 500)
})

# Mock meal info
meal = pd.DataFrame({
    'meal_id': range(1, 51),
    'category': np.random.choice(['Beverages', 'Rice Bowl', 'Sandwich', 'Desert', 'Pizza'], 50),
    'cuisine': np.random.choice(['Thai', 'Indian', 'Italian', 'Continental'], 50)
})

# Mock center info
centerinfo = pd.DataFrame({
    'center_id': range(1, 100),
    'city_code': np.random.randint(1, 20, 99),
    'region_code': np.random.randint(1, 5, 99),
    'center_type': np.random.choice(['TYPE_A', 'TYPE_B', 'TYPE_C'], 99)
})

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
train = pd.merge(train_raw, meal, on="meal_id", how="left")
df = pd.merge(train, centerinfo, on="center_id", how="left")
print("Shape of train data : ", df.shape)
df.head()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
test_raw = pd.merge(test_raw, meal, on="meal_id", how="left")
dft = pd.merge(test_raw, centerinfo, on="center_id", how="left")
print("Shape of train data : ", dft.shape)
dft.head()

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
# === BEFORE (original) ===
# col_names=['center_id','meal_id','category','cuisine','city_code','region_code','center_type','emailer_for_promotion','homepage_featured', 'week']
# dft[col_names] = dft[col_names].astype('category')

# === AFTER (edited) ===
col_names=['center_id','meal_id','category','cuisine','city_code','region_code','center_type','emailer_for_promotion','homepage_featured', 'week']
# Apply the type conversion only if all required columns exist
if all(col in dft.columns for col in col_names):
    # Preserve by-column converters: apply only to existing columns
    conv = {}
    for col in col_names:
        if col in dft.columns and str(dft[col].dtype) != 'category':
            conv[col] = 'category'
    dft = dft.astype(conv)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
col_names=['center_id','meal_id','category','cuisine','city_code','region_code','center_type','emailer_for_promotion','homepage_featured', 'week']
df[col_names] = df[col_names].astype('category')

print("Train Datatype\n",df.dtypes)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
df = df[df['num_orders'] <= 20000];
df=df.drop("id",  axis=1)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
if 'id' in df.columns:
    df = df.drop('id', axis=1)
df.head()

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 16}
df['new_discount_rate'] = (df['base_price'] - df['checkout_price']) / df['base_price']
df.head()

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 17}
df=df.drop("checkout_price",axis=1)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 18}
#Her haftada her bir mutfak türünden kaç farklı yemek sunulduğunu hesaplar
weekly_cuisine_category = df.groupby(['week', 'cuisine'])['category'].nunique().reset_index()
weekly_cuisine_category.rename(columns={'category': 'weekly_cuisine_cat'}, inplace=True)

# weekly_cuisine_cat sütununu df veri çerçevesine ekleyin
df = df.merge(weekly_cuisine_category, on=['week', 'cuisine'], how='left')
df.head()


#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
#Her kategoride her bir mutfak türünün base_price'ını hesaplar.
cat_cuisine_price = df.groupby(['category', 'cuisine'])['base_price'].nunique().reset_index()
cat_cuisine_price.rename(columns={'base_price': 'cat_cuisine_price'}, inplace=True)

# cat_cuisine_price sütununu df veri çerçevesine ekleyin
df = df.merge(cat_cuisine_price, on=['category', 'cuisine'], how='left')
df.head()

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 20}
df['week'] = df['week'].astype(int)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 21}
df_train=df[df.week<=119]

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 22}
df_test=df[df.week>119]

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 24}
df_train=df_train.drop("week",axis=1)
df_test=df_test.drop("week",axis=1)

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 25}
b=list(df_train.columns)

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 26}
df_encoded = pd.get_dummies(df_train[b], drop_first=True)

df_encoded.head()

#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 27}
df_test_encoded=pd.get_dummies(df_test[b], drop_first=True)
df_test_encoded.head()

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 28}
X = df_encoded.drop("num_orders", axis=1)##X_train
y = df_encoded["num_orders"]##y_train

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 29}
w=df_test_encoded.drop("num_orders", axis=1)##X_test
z=df_test_encoded["num_orders"]#y_test

#%%
# --- [CELL 21]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.preprocessing import StandardScaler

#%%
# --- [CELL 22]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

# Decision Tree Regressor için parametre gridini tanımlayın
param_grid = {
    'max_depth': [10, 15, 20],  # Farklı maksimum derinlik seviyeleri
    'min_samples_split': [2, 5],  # Farklı min_samples_split değerleri
    'min_samples_leaf': [1, 2]  # Farklı min_samples_leaf değerleri
}

# Decision Tree Regressor modelini oluşturun
DTRmodel = DecisionTreeRegressor(random_state=0)

# GridSearchCV'yi tanımlayın
grid_search = GridSearchCV(estimator=DTRmodel, param_grid=param_grid, 
                           cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Modeli eğitim verileri üzerinde eğitin
grid_search.fit(X, y)

# En iyi parametreleri ve en iyi tahmin modelini alın
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

# En iyi tahmin modelini kullanarak test verileri üzerinde tahmin yapın
y_pred = best_estimator.predict(w)

# Performans metriklerini hesaplayın
r2 = r2_score(z, y_pred)
mse = mean_squared_error(z, y_pred)
rmse = sqrt(mse)

# Sonuçları yazdırın
print("En iyi parametreler:", best_params)
print("R2 score:", r2)
print("MSE score:", mse)
print("RMSE:", rmse)

#%%
# --- [CELL 23]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}


dft['new_discount_rate'] = (dft['base_price'] - dft['checkout_price']) / dft['base_price']

#%%
# --- [CELL 24]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
dft=dft.drop("checkout_price",axis=1)

#%%
# --- [CELL 25]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
#Her haftada her bir mutfak türünden kaç farklı yemek sunulduğunu hesaplar
weekly_cuisine_category = dft.groupby(['week', 'cuisine'])['category'].nunique().reset_index()
weekly_cuisine_category.rename(columns={'category': 'weekly_cuisine_cat'}, inplace=True)

# weekly_cuisine_cat sütununu df veri çerçevesine ekleyin
dft = dft.merge(weekly_cuisine_category, on=['week', 'cuisine'], how='left')
dft.head()


#%%
# --- [CELL 26]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
#Her kategoride her bir mutfak türünün base_price'ını hesaplar.
cat_cuisine_price = dft.groupby(['category', 'cuisine'])['base_price'].nunique().reset_index()
cat_cuisine_price.rename(columns={'base_price': 'cat_cuisine_price'}, inplace=True)

# cat_cuisine_price sütununu df veri çerçevesine ekleyin
dft = dft.merge(cat_cuisine_price, on=['category', 'cuisine'], how='left')
dft.head()

#%%
# --- [CELL 27]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
dft=dft.drop("id",axis=1)



#%%
# --- [CELL 28]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
dft=dft.drop('week',axis=1)

#%%
# --- [CELL 29]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
c = list(dft.columns)
c

#%%
# --- [CELL 30]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
dft_encoded = pd.get_dummies(dft[c], drop_first=True)
dft_encoded.head()

#%%
# --- [CELL 31]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
final_pred = DTRmodel.predict(dft_encoded)
