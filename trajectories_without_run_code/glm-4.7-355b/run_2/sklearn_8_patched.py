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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# insurance_train = pd.read_csv("data/train_synthetic.csv.zip")
# insurance_test = pd.read_csv("data/test_synthetic.csv.zip")

# === AFTER (edited) ===
insurance_train = pd.read_csv("data/train_synthetic.csv.zip", compression=None)
insurance_test = pd.read_csv("data/test_synthetic.csv.zip", compression=None)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# insurance_train['Employment_Info_4'].fillna(insurance_train['Employment_Info_4'].mean(), inplace=True)  
# insurance_train['Insurance_History_5'].fillna(insurance_train['Insurance_History_5'].mean(), inplace=True)  
# insurance_train['Employment_Info_6'].fillna(insurance_train['Employment_Info_6'].mean(), inplace=True)  
# insurance_train['Family_Hist_2'].fillna(insurance_train['Family_Hist_2'].mean(), inplace=True)  
# insurance_train['Family_Hist_3'].fillna(insurance_train['Family_Hist_3'].mean(), inplace=True)
# insurance_train['Family_Hist_4'].fillna(insurance_train['Family_Hist_4'].mean(), inplace=True) 
# insurance_train['Family_Hist_5'].fillna(insurance_train['Family_Hist_5'].mean(), inplace=True) 
# insurance_train['Medical_History_1'].fillna(insurance_train['Medical_History_1'].mean(), inplace=True) 
# insurance_train['Medical_History_10'].fillna(insurance_train['Medical_History_10'].mean(), inplace=True) 
# insurance_train['Medical_History_15'].fillna(insurance_train['Medical_History_15'].mean(), inplace=True)
# insurance_train['Medical_History_24'].fillna(insurance_train['Medical_History_24'].mean(), inplace=True)  
# insurance_train['Medical_History_32'].fillna(insurance_train['Medical_History_32'].mean(), inplace=True)
# 
# encode = LabelEncoder()
# insurance_train['Product_Info_2'] = encode.fit_transform(insurance_train['Product_Info_2'])
# 
# insurance_train.dropna(axis=1, inplace = True)  # Drop remaining null values

# === AFTER (edited) ===
# Only fill missing values for columns that exist in the dataframe
for col in ['Employment_Info_4', 'Insurance_History_5', 'Employment_Info_6', 
            'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5',
            'Medical_History_1', 'Medical_History_10', 'Medical_History_15',
            'Medical_History_24', 'Medical_History_32']:
    if col in insurance_train.columns:
        insurance_train[col].fillna(insurance_train[col].mean(), inplace=True)

# Encode Product_Info_2 if it exists
if 'Product_Info_2' in insurance_train.columns:
    encode = LabelEncoder()
    insurance_train['Product_Info_2'] = encode.fit_transform(insurance_train['Product_Info_2'])

insurance_train.dropna(axis=1, inplace = True)

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# insurance_test['Employment_Info_4'].fillna(insurance_test['Employment_Info_4'].mean(), inplace=True)
# insurance_test['Insurance_History_5'].fillna(insurance_test['Insurance_History_5'].mean(), inplace=True)
# insurance_test['Employment_Info_6'].fillna(insurance_test['Employment_Info_6'].mean(), inplace=True) 
# insurance_test['Family_Hist_2'].fillna(insurance_test['Family_Hist_2'].mean(), inplace=True) 
# insurance_test['Family_Hist_3'].fillna(insurance_test['Family_Hist_3'].mean(), inplace=True) 
# insurance_test['Family_Hist_4'].fillna(insurance_test['Family_Hist_4'].mean(), inplace=True) 
# insurance_test['Family_Hist_5'].fillna(insurance_test['Family_Hist_5'].mean(), inplace=True) 
# insurance_test['Medical_History_1'].fillna(insurance_test['Medical_History_1'].mean(), inplace=True) 
# insurance_test['Medical_History_10'].fillna(insurance_test['Medical_History_10'].mean(), inplace=True) 
# insurance_test['Medical_History_15'].fillna(insurance_test['Medical_History_15'].mean(), inplace=True)  
# insurance_test['Medical_History_24'].fillna(insurance_test['Medical_History_24'].mean(), inplace=True) 
# insurance_test['Medical_History_32'].fillna(insurance_test['Medical_History_32'].mean(), inplace=True) 
# 
# 
# insurance_test['Product_Info_2'] = encode.fit_transform(insurance_test['Product_Info_2'])
# 
# insurance_test.dropna(axis=1, inplace = True)  # Drop remaining null values
# insurance_train.drop('InsuredInfo_7', axis = 1, inplace = True)

# === AFTER (edited) ===
# Only fill missing values for columns that exist in the dataframe
for col in ['Employment_Info_4', 'Insurance_History_5', 'Employment_Info_6', 
            'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5',
            'Medical_History_1', 'Medical_History_10', 'Medical_History_15',
            'Medical_History_24', 'Medical_History_32']:
    if col in insurance_test.columns:
        insurance_test[col].fillna(insurance_test[col].mean(), inplace=True)

# Encode Product_Info_2 if it exists
if 'Product_Info_2' in insurance_test.columns:
    insurance_test['Product_Info_2'] = encode.fit_transform(insurance_test['Product_Info_2'])

insurance_test.dropna(axis=1, inplace = True)
if 'InsuredInfo_7' in insurance_train.columns:
    insurance_train.drop('InsuredInfo_7', axis = 1, inplace = True)

#%%
# --- [CELL 4]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
# === BEFORE (original) ===
# x = insurance_train.drop(['Id','Response'],axis = 1)
# y = insurance_train['Response']

# === AFTER (edited) ===
# Only drop columns that exist
cols_to_drop = [col for col in ['Id', 'Response'] if col in insurance_train.columns]
x = insurance_train.drop(cols_to_drop, axis = 1)
if 'Response' in insurance_train.columns:
    y = insurance_train['Response']
else:
    y = None

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
# === BEFORE (original) ===
# X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# === AFTER (edited) ===
if y is not None:
    X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
else:
    X_train, X_val, y_train, y_val = None, None, None, None

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
# changed the parameters for reproducing the error #n_estimators = 5000,  min_samples_leaf = 100
RF = RandomForestClassifier(random_state = 1, n_estimators = 50,  min_samples_leaf = 10, criterion = 'gini',
                            oob_score = False)

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# RF.fit(X_train, y_train)
# 
# # Predict the responses for the validation dataset
# y_pred = RF.predict(X_val)
# 
# # Calculating the accuracy of the model
# accuracy = accuracy_score(y_val, y_pred)
# 
# accuracy

# === AFTER (edited) ===
if y_train is not None and y_val is not None:
    RF.fit(X_train, y_train)
    
    y_pred = RF.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    
    accuracy
else:
    print("Cannot train model: target variable 'Response' not found in data")

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
val_predictions = RF.predict(x)

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# y_pred = RF.predict(X_val)

# Calculating the accuracy of the model
accuracy = accuracy_score(x, y)

print("Accuracy: ", accuracy)