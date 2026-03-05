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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
train_raw = pd.read_csv('data/train.csv')
test_raw = pd.read_csv('data/test.csv')
meal = pd.read_csv('data/meal_info.csv')
centerinfo = pd.read_csv('data/fulfilment_center_info.csv')


#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# train = pd.merge(train_raw, meal, on="meal_id", how="left")
# df = pd.merge(train, centerinfo, on="center_id", how="left")
# print("Shape of train data : ", df.shape)
# df.head()

# === AFTER (edited) ===
# Check column names to identify the correct merge key
print("train_raw columns:", train_raw.columns.tolist())
print("meal columns:", meal.columns.tolist())

# Find the matching column (case-insensitive)
train_cols_lower = [c.lower() for c in train_raw.columns]
meal_cols_lower = [c.lower() for c in meal.columns]

# Try to find meal_id (case-insensitive)
meal_col_in_train = None
meal_col_in_meal = None

for i, col in enumerate(train_raw.columns):
    if col.lower() == 'meal_id':
        meal_col_in_train = col
        break

for i, col in enumerate(meal.columns):
    if col.lower() == 'meal_id':
        meal_col_in_meal = col
        break

print(f"Found 'meal_id' in train_raw as: {meal_col_in_train}")
print(f"Found 'meal_id' in meal as: {meal_col_in_meal}")

if meal_col_in_train and meal_col_in_meal:
    train = pd.merge(train_raw, meal, left_on=meal_col_in_train, right_on=meal_col_in_meal, how='left')
else:
    # Fallback to original if found
    if 'meal_id' in train_raw.columns and 'meal_id' in meal.columns:
        train = pd.merge(train_raw, meal, on="meal_id", how="left")

# Check center_id column names
center_col_in_train = None
center_col_in_center = None

for i, col in enumerate(train.columns):
    if col.lower() == 'center_id':
        center_col_in_train = col
        break

for i, col in enumerate(centerinfo.columns):
    if col.lower() == 'center_id':
        center_col_in_center = col
        break

print(f"Found 'center_id' in train as: {center_col_in_train}")
print(f"Found 'center_id' in centerinfo as: {center_col_in_center}")

if center_col_in_train and center_col_in_center:
    df = pd.merge(train, centerinfo, left_on=center_col_in_train, right_on=center_col_in_center, how='left')
else:
    # Fallback to original if found
    if 'center_id' in train.columns and 'center_id' in centerinfo.columns:
        df = pd.merge(train, centerinfo, on="center_id", how="left")

print("Shape of train data : ", df.shape)
df.head()

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
test_raw = pd.merge(test_raw, meal, on="meal_id", how="left")
dft = pd.merge(test_raw, centerinfo, on="center_id", how="left")
print("Shape of train data : ", dft.shape)
dft.head()

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
col_names=['center_id','meal_id','category','cuisine','city_code','region_code','center_type','emailer_for_promotion','homepage_featured', 'week']
dft[col_names] = dft[col_names].astype('category')

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
col_names=['center_id','meal_id','category','cuisine','city_code','region_code','center_type','emailer_for_promotion','homepage_featured', 'week']
df[col_names] = df[col_names].astype('category')

print("Train Datatype\n",df.dtypes)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df = df[df['num_orders'] <= 20000];
df=df.drop("id",  axis=1)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
if 'id' in df.columns:
    df = df.drop('id', axis=1)
df.head()

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df['new_discount_rate'] = (df['base_price'] - df['checkout_price']) / df['base_price']
df.head()

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df=df.drop("checkout_price",axis=1)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
#Her haftada her bir mutfak türünden kaç farklı yemek sunulduğunu hesaplar
weekly_cuisine_category = df.groupby(['week', 'cuisine'])['category'].nunique().reset_index()
weekly_cuisine_category.rename(columns={'category': 'weekly_cuisine_cat'}, inplace=True)

# weekly_cuisine_cat sütununu df veri çerçevesine ekleyin
df = df.merge(weekly_cuisine_category, on=['week', 'cuisine'], how='left')
df.head()


#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
#Her kategoride her bir mutfak türünün base_price'ını hesaplar.
cat_cuisine_price = df.groupby(['category', 'cuisine'])['base_price'].nunique().reset_index()
cat_cuisine_price.rename(columns={'base_price': 'cat_cuisine_price'}, inplace=True)

# cat_cuisine_price sütununu df veri çerçevesine ekleyin
df = df.merge(cat_cuisine_price, on=['category', 'cuisine'], how='left')
df.head()

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df['week'] = df['week'].astype(int)

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df_train=df[df.week<=119]

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df_test=df[df.week>119]

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df_train=df_train.drop("week",axis=1)
df_test=df_test.drop("week",axis=1)

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
b=list(df_train.columns)

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df_encoded = pd.get_dummies(df_train[b], drop_first=True)

df_encoded.head()

#%%
# --- [CELL 18]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
df_test_encoded=pd.get_dummies(df_test[b], drop_first=True)
df_test_encoded.head()

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
X = df_encoded.drop("num_orders", axis=1)##X_train
y = df_encoded["num_orders"]##y_train

#%%
# --- [CELL 20]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
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
