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
import os
import pandas as pd
import numpy as np

# Since the files are Git LFS pointers (not actual data), create sample data
np.random.seed(42)

# Create sample training data
n_train = 1000
train_data = {
    'Id': range(n_train),
    'Response': np.random.randint(1, 9, n_train),
    'Product_Info_1': np.random.randint(1, 10, n_train),
    'Product_Info_2': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1', 'C2'], n_train),
    'Product_Info_3': np.random.randint(1, 30, n_train),
    'InsuredInfo_1': np.random.randint(1, 3, n_train),
    'InsuredInfo_2': np.random.randint(1, 4, n_train),
    'InsuredInfo_3': np.random.randint(1, 3, n_train),
    'InsuredInfo_4': np.random.randint(1, 4, n_train),
    'InsuredInfo_5': np.random.randint(1, 3, n_train),
    'InsuredInfo_6': np.random.randint(1, 4, n_train),
    'InsuredInfo_7': np.random.randint(1, 3, n_train),
    'Employment_Info_1': np.random.randint(1, 3, n_train),
    'Employment_Info_2': np.random.uniform(0, 1, n_train),
    'Employment_Info_3': np.random.randint(1, 4, n_train),
    'Employment_Info_4': np.random.uniform(0, 1, n_train),
    'Employment_Info_5': np.random.randint(1, 5, n_train),
    'Employment_Info_6': np.random.uniform(0, 1, n_train),
    'Insurance_History_1': np.random.randint(1, 4, n_train),
    'Insurance_History_2': np.random.uniform(0, 1, n_train),
    'Insurance_History_3': np.random.randint(1, 4, n_train),
    'Insurance_History_4': np.random.uniform(0, 1, n_train),
    'Insurance_History_5': np.random.uniform(0, 1, n_train),
    'Ins_History_7': np.random.randint(1, 5, n_train),
    'Ins_History_8': np.random.uniform(0, 1, n_train),
    'Ins_History_9': np.random.uniform(0, 1, n_train),
    'Family_Hist_1': np.random.uniform(0, 1, n_train),
    'Family_Hist_2': np.random.uniform(0, 1, n_train),
    'Family_Hist_3': np.random.uniform(0, 1, n_train),
    'Family_Hist_4': np.random.uniform(0, 1, n_train),
    'Family_Hist_5': np.random.uniform(0, 1, n_train),
    'Medical_History_1': np.random.uniform(0, 1, n_train),
    'Medical_History_2': np.random.randint(1, 10, n_train),
    'Medical_History_3': np.random.randint(1, 10, n_train),
    'Medical_History_4': np.random.randint(1, 10, n_train),
    'Medical_History_5': np.random.randint(1, 10, n_train),
    'Medical_History_6': np.random.randint(1, 10, n_train),
    'Medical_History_7': np.random.randint(1, 10, n_train),
    'Medical_History_8': np.random.randint(1, 10, n_train),
    'Medical_History_9': np.random.randint(1, 10, n_train),
    'Medical_History_10': np.random.uniform(0, 1, n_train),
    'Medical_History_11': np.random.randint(1, 10, n_train),
    'Medical_History_12': np.random.randint(1, 10, n_train),
    'Medical_History_13': np.random.randint(1, 10, n_train),
    'Medical_History_14': np.random.randint(1, 10, n_train),
    'Medical_History_15': np.random.uniform(0, 1, n_train),
    'Medical_History_16': np.random.randint(1, 10, n_train),
    'Medical_History_17': np.random.randint(1, 10, n_train),
    'Medical_History_18': np.random.randint(1, 10, n_train),
    'Medical_History_19': np.random.randint(1, 10, n_train),
    'Medical_History_20': np.random.randint(1, 10, n_train),
    'Medical_History_21': np.random.randint(1, 10, n_train),
    'Medical_History_22': np.random.randint(1, 10, n_train),
    'Medical_History_23': np.random.randint(1, 10, n_train),
    'Medical_History_24': np.random.uniform(0, 1, n_train),
    'Medical_History_25': np.random.randint(1, 10, n_train),
    'Medical_History_26': np.random.randint(1, 10, n_train),
    'Medical_History_27': np.random.randint(1, 10, n_train),
    'Medical_History_28': np.random.randint(1, 10, n_train),
    'Medical_History_29': np.random.randint(1, 10, n_train),
    'Medical_History_30': np.random.randint(1, 10, n_train),
    'Medical_History_31': np.random.randint(1, 10, n_train),
    'Medical_History_32': np.random.uniform(0, 1, n_train),
    'Medical_History_33': np.random.randint(1, 10, n_train),
    'Medical_History_34': np.random.randint(1, 10, n_train),
    'Medical_History_35': np.random.randint(1, 10, n_train),
    'Medical_History_36': np.random.randint(1, 10, n_train),
    'Medical_History_37': np.random.randint(1, 10, n_train),
    'Medical_History_38': np.random.randint(1, 10, n_train),
    'Medical_History_39': np.random.randint(1, 10, n_train),
    'Medical_History_40': np.random.randint(1, 10, n_train),
    'Medical_History_41': np.random.randint(1, 10, n_train)
}

insurance_train = pd.DataFrame(train_data)

# Create sample test data
n_test = 200
test_data = {k: v[:n_test] if k != 'Id' else range(n_test) for k, v in train_data.items() if k != 'Response'}
insurance_test = pd.DataFrame(test_data)

# Introduce some NaN values for testing filler logic
for col in ['Employment_Info_4', 'Insurance_History_5', 'Employment_Info_6', 'Family_Hist_2', 
            'Family_Hist_3', 'Family_Hist_4', 'Family_Hist_5', 'Medical_History_1', 
            'Medical_History_10', 'Medical_History_15', 'Medical_History_24', 'Medical_History_32']:
    mask = np.random.random(n_train) < 0.1
    insurance_train.loc[mask, col] = np.nan
    mask = np.random.random(n_test) < 0.1
    insurance_test.loc[mask, col] = np.nan

print(f"Training data shape: {insurance_train.shape}")
print(f"Test data shape: {insurance_test.shape}")

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