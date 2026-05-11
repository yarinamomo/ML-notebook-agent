# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
df = pd.read_csv('data/training.csv')
df.info()
df

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
columns_to_remove = ['PurchDate', 'VehYear', 'WheelTypeID', 'BYRNO', 'VNZIP1', 'Model', 'Trim', 'VNST', 'SubModel']
df

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
target = df['IsBadBuy']
inputs = df.drop(columns=['IsBadBuy'])

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(inputs, target, test_size=0.2, random_state=1)
inputs=x_train

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
column_ranges={
    'VehicleAge': (0,30),
    'VehOdo': (0,120000),
    'MMRAcquisitionAuctionAveragePrice': (800,46000),
    'MMRAcquisitionAuctionCleanPrice': (1000,46000),
    'MMRAcquisitionRetailAveragePrice': (1000,46000),
    'MMRAcquisitonRetailCleanPrice': (1000,46000),
    'MMRCurrentAuctionAveragePrice': (300,46000), 
    'MMRCurrentAuctionCleanPrice': (400,46000), 
    'MMRCurrentRetailAveragePrice': (800,46000),
    'MMRCurrentRetailCleanPrice': (1000,46000),
    'VehBCost': (1000,46000),                         
    'WarrantyCost': (400,8000) 
}
for column, (min_val, max_val) in column_ranges.items():
    out_of_range_count = ((inputs[column] < min_val) | (inputs[column] > max_val)).sum()
    print(f"The number of out-of-range cells in '{column}' variable is: {out_of_range_count}")

for column , (min_val,max_val) in column_ranges.items():
    inputs[column]=inputs[column].apply(lambda x:x if min_val<=x<=max_val else None)
#print(inputs)
inputs.describe()

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
import numpy as np
def frequency_table(variable):
    unique_elements,counts=np.unique(variable.dropna(),return_counts=True)
    percentage=(counts/len(variable)*100)
    for i , j , k in zip(unique_elements,counts,percentage):
        print(f"{i} :count{j} , percentage: {k:.2f}")
    return
categorical_fields = [
    "Auction",
    "Make",
    "Color",
    "Transmission",
    "WheelType",
    "Nationality",
    "Size",
    "TopThreeAmericanName",
    "PRIMEUNIT",
    "AUCGUART"
]
continuous_fields = [col for col in inputs.columns if col not in categorical_fields]
for col in categorical_fields:
    print(f"Frequency Table for {col}:")
    frequency_table(inputs[col])
    print("-" * 40)

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
inputs["Transmission"]=inputs["Transmission"].replace("Manual",'MANUAL')
frequency_table(inputs["Transmission"])
print("-"*40)
inputs["Color"].info()
print("-"*40)
inputs["Color"]=inputs["Color"].replace("NOT AVAIL",np.nan)
inputs["Color"].info()

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
def replace_rare_classes(df, column, threshold=1):
    unique_elements, counts = np.unique(df[column].dropna(), return_counts=True)
    percentage = (counts / len(df[column])) * 100
    rare_classes = [elem for elem, pct in zip(unique_elements, percentage) if pct < threshold]
    return df[column].replace(rare_classes, 'OTHER')

inputs['Make'] = replace_rare_classes(inputs, 'Make', threshold=1)
print("\nUpdated Frequency Table for 'Make':")
frequency_table(inputs['Make'])


inputs['Color'] = replace_rare_classes(inputs, 'Color', threshold=1)
print("\nUpdated Frequency Table for 'Color':")
frequency_table(inputs['Color'])

#%%
# --- [CELL 8]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
# === BEFORE (original) ===
# print(f"Number of continuous fields before feature screening: {len(continuous_fields)}")
# print(f"Number of categorical fields before feature screening: {len(categorical_fields)}")
# 
# min_cv=0.1
# cv_values=inputs[continuous_fields].std()/inputs[continuous_fields].mean()
# selected_columns=cv_values[cv_values<0.1].index
# print(f"\nNumber of Features with a coefficient of variation less than **0.1**: {len(selected_columns)}")
# filtered_con=inputs[selected_columns]
# inputs_con=inputs[continuous_fields].drop(selected_columns,axis=1)
# 
# threshold=99
# mode_category=(inputs[categorical_fields].apply(lambda x:x.value_counts().max()/len(x)*100))
# selected_categorical_columns=mode_category[mode_category>threshold].index
# print(f"Number of features where the mode category percentage is greater than **99%**: {len(selected_columns)}")
# mode_filtered_inputs=inputs[selected_categorical_columns]
# inputs_cat=inputs[categorical_fields].drop(selected_categorical_columns,axis=1)
# 
# threshold=90
# distinct_percentage=inputs_cat[categorical_fields].apply(lambda x:x.dropna().nunique()/x.count()*100)
# selected_categorical_columns=distinct_percentage[distinct_percentage>threshold].index
# print(f"Number of Features with a percentage of unique categories exceeding **90%**: {len(selected_columns)}")
# distinct_filtered_inputs=inputs_cat[selected_categorical_columns]
# inputs_cat=inputs[categorical_fields].drop(selected_categorical_columns,axis=1)
# 
# filtered_df=pd.concat([inputs_con,inputs_cat,target],axis=1)
# 
# print(f"\nNumber of continuous fields AFTER feature screening: {len(continuous_fields)}")
# print(f"Number of categorical fields AFTER feature screening: {len(categorical_fields)}")

# === AFTER (edited) ===
print(f"Number of continuous fields before feature screening: {len(continuous_fields)}")
print(f"Number of categorical fields before feature screening: {len(categorical_fields)}")

# Recompute field groups from current inputs to avoid stale lists and type issues
continuous_fields = inputs.select_dtypes(include=[np.number]).columns.tolist()
categorical_fields = [col for col in inputs.columns if col not in continuous_fields]

min_cv = 0.1
cv_values = inputs[continuous_fields].std() / inputs[continuous_fields].mean()
selected_columns = cv_values[cv_values < min_cv].index
print(f"\nNumber of Features with a coefficient of variation less than **0.1**: {len(selected_columns)}")
filtered_con = inputs[selected_columns]
inputs_con = inputs[continuous_fields].drop(columns=selected_columns, axis=1)

threshold = 99
mode_category = inputs[categorical_fields].apply(lambda x: x.value_counts().max() / len(x) * 100)
selected_categorical_columns = mode_category[mode_category > threshold].index
print(f"Number of features where the mode category percentage is greater than **99%**: {len(selected_categorical_columns)}")
mode_filtered_inputs = inputs[selected_categorical_columns]
inputs_cat = inputs[categorical_fields].drop(columns=selected_categorical_columns, axis=1)

threshold = 90
distinct_percentage = inputs_cat.apply(lambda x: x.dropna().nunique() / x.count() * 100)
selected_categorical_columns = distinct_percentage[distinct_percentage > threshold].index
print(f"Number of Features with a percentage of unique categories exceeding **90%**: {len(selected_categorical_columns)}")
distinct_filtered_inputs = inputs_cat[selected_categorical_columns]
inputs_cat = inputs_cat.drop(columns=selected_categorical_columns, axis=1)

filtered_df = pd.concat([inputs_con, inputs_cat, target], axis=1)

print(f"\nNumber of continuous fields AFTER feature screening: {len(inputs_con.columns)}")
print(f"Number of categorical fields AFTER feature screening: {len(inputs_cat.columns)}")