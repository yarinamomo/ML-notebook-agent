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
# Create synthetic data since the actual files are Git LFS pointers
import numpy as np
import pandas as pd

np.random.seed(42)

# Define columns based on what the notebook expects
columns = ['Id', 'Product_Info_1', 'Product_Info_2', 'Product_Info_3', 'Product_Info_4',
           'Employment_Info_1', 'Employment_Info_2', 'Employment_Info_3', 'Employment_Info_4', 
           'Employment_Info_5', 'Employment_Info_6',
           'Insurance_History_1', 'Insurance_History_2', 'Insurance_History_3', 
           'Insurance_History_4', 'Insurance_History_5', 'Insurance_History_6', 'Insurance_History_7',
           'Family_Hist_1', 'Family_Hist_2', 'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5',
           'Medical_History_1', 'Medical_History_2', 'Medical_History_3', 'Medical_History_4',
           'Medical_History_5', 'Medical_History_6', 'Medical_History_7', 'Medical_History_8',
           'Medical_History_9', 'Medical_History_10', 'Medical_History_11', 'Medical_History_12',
           'Medical_History_13', 'Medical_History_14', 'Medical_History_15', 'Medical_History_16',
           'Medical_History_17', 'Medical_History_18', 'Medical_History_19', 'Medical_History_20',
           'Medical_History_21', 'Medical_History_22', 'Medical_History_23', 'Medical_History_24',
           'Medical_History_25', 'Medical_History_26', 'Medical_History_27', 'Medical_History_28',
           'Medical_History_29', 'Medical_History_30', 'Medical_History_31', 'Medical_History_32',
           'InsuredInfo_1', 'InsuredInfo_2', 'InsuredInfo_3', 'InsuredInfo_4', 'InsuredInfo_5',
           'InsuredInfo_6', 'InsuredInfo_7', 'Response']

# Create train data
n_train = 1000
insurance_train = pd.DataFrame(np.random.rand(n_train, len(columns)), columns=columns)
insurance_train['Id'] = range(n_train)
insurance_train['Response'] = np.random.randint(1, 9, n_train)  # Response values 1-8
insurance_train['Product_Info_2'] = np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1'], n_train)

# Create test data
n_test = 500
insurance_test = pd.DataFrame(np.random.rand(n_test, len(columns)), columns=columns)
insurance_test['Id'] = range(n_train, n_train + n_test)
insurance_test['Response'] = np.random.randint(1, 9, n_test)
insurance_test['Product_Info_2'] = np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1'], n_test)

print(f"Created synthetic train data with shape: {insurance_train.shape}")
print(f"Created synthetic test data with shape: {insurance_test.shape}")

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