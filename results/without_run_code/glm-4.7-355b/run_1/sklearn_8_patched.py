# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
insurance_train = pd.read_csv("data/train_synthetic.csv.zip")
insurance_test = pd.read_csv("data/test_synthetic.csv.zip")

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
insurance_train['Employment_Info_4'].fillna(insurance_train['Employment_Info_4'].mean(), inplace=True)  
insurance_train['Insurance_History_5'].fillna(insurance_train['Insurance_History_5'].mean(), inplace=True)  
insurance_train['Employment_Info_6'].fillna(insurance_train['Employment_Info_6'].mean(), inplace=True)  
insurance_train['Family_Hist_2'].fillna(insurance_train['Family_Hist_2'].mean(), inplace=True)  
insurance_train['Family_Hist_3'].fillna(insurance_train['Family_Hist_3'].mean(), inplace=True)
insurance_train['Family_Hist_4'].fillna(insurance_train['Family_Hist_4'].mean(), inplace=True) 
insurance_train['Family_Hist_5'].fillna(insurance_train['Family_Hist_5'].mean(), inplace=True) 
insurance_train['Medical_History_1'].fillna(insurance_train['Medical_History_1'].mean(), inplace=True) 
insurance_train['Medical_History_10'].fillna(insurance_train['Medical_History_10'].mean(), inplace=True) 
insurance_train['Medical_History_15'].fillna(insurance_train['Medical_History_15'].mean(), inplace=True)
insurance_train['Medical_History_24'].fillna(insurance_train['Medical_History_24'].mean(), inplace=True)  
insurance_train['Medical_History_32'].fillna(insurance_train['Medical_History_32'].mean(), inplace=True)

encode = LabelEncoder()
insurance_train['Product_Info_2'] = encode.fit_transform(insurance_train['Product_Info_2'])

insurance_train.dropna(axis=1, inplace = True)  # Drop remaining null values

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
insurance_test['Employment_Info_4'].fillna(insurance_test['Employment_Info_4'].mean(), inplace=True)
insurance_test['Insurance_History_5'].fillna(insurance_test['Insurance_History_5'].mean(), inplace=True)
insurance_test['Employment_Info_6'].fillna(insurance_test['Employment_Info_6'].mean(), inplace=True) 
insurance_test['Family_Hist_2'].fillna(insurance_test['Family_Hist_2'].mean(), inplace=True) 
insurance_test['Family_Hist_3'].fillna(insurance_test['Family_Hist_3'].mean(), inplace=True) 
insurance_test['Family_Hist_4'].fillna(insurance_test['Family_Hist_4'].mean(), inplace=True) 
insurance_test['Family_Hist_5'].fillna(insurance_test['Family_Hist_5'].mean(), inplace=True) 
insurance_test['Medical_History_1'].fillna(insurance_test['Medical_History_1'].mean(), inplace=True) 
insurance_test['Medical_History_10'].fillna(insurance_test['Medical_History_10'].mean(), inplace=True) 
insurance_test['Medical_History_15'].fillna(insurance_test['Medical_History_15'].mean(), inplace=True)  
insurance_test['Medical_History_24'].fillna(insurance_test['Medical_History_24'].mean(), inplace=True) 
insurance_test['Medical_History_32'].fillna(insurance_test['Medical_History_32'].mean(), inplace=True) 


insurance_test['Product_Info_2'] = encode.fit_transform(insurance_test['Product_Info_2'])

insurance_test.dropna(axis=1, inplace = True)  # Drop remaining null values
insurance_train.drop('InsuredInfo_7', axis = 1, inplace = True)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
x = insurance_train.drop(['Id','Response'],axis = 1)
y = insurance_train['Response']

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# changed the parameters for reproducing the error #n_estimators = 5000,  min_samples_leaf = 100
RF = RandomForestClassifier(random_state = 1, n_estimators = 50,  min_samples_leaf = 10, criterion = 'gini',
                            oob_score = False)

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
RF.fit(X_train, y_train)

# Predict the responses for the validation dataset
y_pred = RF.predict(X_val)

# Calculating the accuracy of the model
accuracy = accuracy_score(y_val, y_pred)

accuracy

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
val_predictions = RF.predict(x)

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# # y_pred = RF.predict(X_val)
# 
# # Calculating the accuracy of the model
# accuracy = accuracy_score(x, y)
# 
# print("Accuracy: ", accuracy)

# === AFTER (edited) ===
accuracy = accuracy_score(val_predictions, y)

print("Accuracy: ", accuracy)