# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# # ===============================================================
# #  CSIRO Biomass 
# # ===============================================================
# 
# # Ignore warnings
# import warnings
# warnings.filterwarnings("ignore")
# 
# # Imports
# import os
# import numpy as np
# import pandas as pd
# from PIL import Image
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import r2_score
# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.models import Model
# 
# # ===============================================================
# # Load CSVs
# # ===============================================================
# train = pd.read_csv("data/train.csv")
# test = pd.read_csv("data/test.csv")
# 
# # ===============================================================
# # Simple Image Feature Extractor (EfficientNet)
# # ===============================================================
# base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
# inp = tf.keras.Input(shape=(224,224,3))
# x = preprocess_input(inp)
# out = base_model(x)
# img_model = Model(inp, out)
# 
# def extract_features(img_path):
#     try:
#         img = Image.open(f"data/{img_path}").convert("RGB")
#         img = img.resize((224,224))
#         arr = np.expand_dims(np.array(img),0)
#         return img_model.predict(arr)[0]
#     except:
#         return np.zeros((img_model.output_shape[-1],))
# 
# # Precompute image features
# train_imgs = np.vstack(train["image_path"].apply(extract_features).values)
# test_imgs  = np.vstack(test["image_path"].apply(extract_features).values)
# 
# # ===============================================================
# # Combine with Tabular Features
# # ===============================================================
# # Drop target columns in train (we predict them later)
# targets = train[["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"]]
# train = train.drop(["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"],axis=1)
# 
# # Numeric
# num_cols = ["Height_Ave_cm","Pre_GSHH_NDVI"]
# # Categorical
# cat_cols = ["State","Species"]
# 
# # Preprocess
# preprocessor = ColumnTransformer([
#     ("num", "passthrough", num_cols),
#     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
# ])
# 
# # Augment with image features
# X_train = np.hstack([train_imgs, train[num_cols].values])
# X_test  = np.hstack([test_imgs, test[num_cols].values])
# 
# # ===============================================================
# # Train Test Split
# # ===============================================================
# X_tr, X_val, y_tr, y_val = train_test_split(X_train, targets.values, test_size=0.15, random_state=42)
# 
# # ===============================================================
# # Model per target
# # ===============================================================
# preds = {}
# for i,col in enumerate(targets.columns):
#     model = GradientBoostingRegressor(random_state=42)
#     model.fit(X_tr, y_tr[:,i])
#     vpred = model.predict(X_val)
#     print(f"{col} R2:", r2_score(y_val[:,i], vpred))
#     preds[col] = model
# 
# # ===============================================================
# # Train on full train and predict test
# # ===============================================================
# submission = pd.DataFrame({"sample_id": test["sample_id"]})
# 
# for col,model in preds.items():
#     submission[col] = model.fit(X_train, targets[col].values).predict(X_test)
# 
# # Melt to long format
# sub_long = submission.melt(id_vars="sample_id",var_name="target_name",value_name="target")
# sub_long.to_csv("submission.csv",index=False)
# 
# print("✅ submission.csv created")
# submission.head()

# === AFTER (edited) ===
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
inp = tf.keras.Input(shape=(224,224,3))
x = preprocess_input(inp)
out = base_model(x)
img_model = Model(inp, out)

def extract_features(img_path):
    try:
        img = Image.open(f"data/{img_path}").convert("RGB")
        img = img.resize((224,224))
        arr = np.expand_dims(np.array(img),0)
        return img_model.predict(arr)[0]
    except:
        return np.zeros((img_model.output_shape[-1],))

train_imgs = np.vstack(train["image_path"].apply(extract_features).values)
test_imgs  = np.vstack(test["image_path"].apply(extract_features).values)

# Detect target columns from train data - columns present in train but not in test
# Then filter to only include numeric columns that can be converted to float
potential_target_cols = [col for col in train.columns if col not in test.columns]
print(f"All columns in train: {train.columns.tolist()}")
print(f"Potential target columns (in train not in test): {potential_target_cols}")

# Filter to only include numeric columns from potential targets
target_cols = [col for col in potential_target_cols if pd.api.types.is_numeric_dtype(train[col])]
print(f"Numeric target columns: {target_cols}")

targets = train[target_cols].astype(float)  # Ensure targets are numeric

# Identify numerical and categorical columns BEFORE dropping target columns
numeric_cols_all = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [col for col in numeric_cols_all if col not in ["image_path", "sample_id"] + target_cols]
print(f"Numeric columns: {num_cols}")

cat_cols_candidate = ["State", "Species"]
# Filter to only keep categorical columns that exist in BOTH train and test
cat_cols = [col for col in cat_cols_candidate if col in train.columns and col in test.columns]
print(f"Categorical columns in both train and test: {cat_cols}")

# Now drop target columns from train
train = train.drop(target_cols, axis=1)

# Build features from columns
train_features = []
test_features = []

if num_cols:
    train_features.append(train[num_cols].values)
    test_features.append(test[num_cols].values)

cat_encoder = OneHotEncoder(handle_unknown="ignore")
if cat_cols:
    train_cat = cat_encoder.fit_transform(train[cat_cols]).toarray()
    test_cat = cat_encoder.transform(test[cat_cols]).toarray()
    train_features.append(train_cat)
    test_features.append(test_cat)

# Combine all features with image features
if train_features:
    train_tabular = np.hstack(train_features)
    test_tabular = np.hstack(test_features)
else:
    train_tabular = np.zeros((train_imgs.shape[0], 0))
    test_tabular = np.zeros((test_imgs.shape[0], 0))

X_train = np.hstack([train_imgs, train_tabular])
X_test  = np.hstack([test_imgs, test_tabular])

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

X_tr, X_val, y_tr, y_val = train_test_split(X_train, targets.values, test_size=0.15, random_state=42)

preds = {}
for i, col in enumerate(targets.columns):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr[:,i])
    vpred = model.predict(X_val)
    print(f"{col} R2:", r2_score(y_val[:,i], vpred))
    preds[col] = model

submission = pd.DataFrame({"sample_id": test["sample_id"]})

for col, model in preds.items():
    submission[col] = model.fit(X_train, targets[col].values).predict(X_test)

# Use a different value_name to avoid conflict
sub_long = submission.melt(id_vars="sample_id", var_name="target_name", value_name="predicted_value")
sub_long.to_csv("submission.csv", index=False)

print("✅ submission.csv created")
submission.head()