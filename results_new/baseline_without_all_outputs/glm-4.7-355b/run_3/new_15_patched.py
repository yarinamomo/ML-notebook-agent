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
# execution_status: {'status': 'error', 'done': True, 'execution_count': 2}
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


# Read data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


# Create feature extraction model
base_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
img_model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    try:
        img = Image.open(f"data/{img_path}").convert("RGB")
        img = img.resize((224,224))
        arr = np.array(img)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, 0)
        features = img_model.predict(arr, verbose=0)
        return features.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return np.zeros((img_model.output_shape[-1],))


# Extract features
print("Extracting training image features...")
train_imgs = np.vstack(train["image_path"].apply(extract_features).values)
print("Extracting test image features...")
test_imgs  = np.vstack(test["image_path"].apply(extract_features).values)


targets = train[["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"]]
train = train.drop(["Dry_Green_g","Dry_Dead_g","Dry_Clover_g","GDM_g","Dry_Total_g"],axis=1)


num_cols = ["Height_Ave_cm","Pre_GSHH_NDVI"]

cat_cols = ["State","Species"]


preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])


X_train = np.hstack([train_imgs, train[num_cols].values])
X_test  = np.hstack([test_imgs, test[num_cols].values])


X_tr, X_val, y_tr, y_val = train_test_split(X_train, targets.values, test_size=0.15, random_state=42)


preds = {}
for i,col in enumerate(targets.columns):
    print(f"Training model for {col}...")
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_tr, y_tr[:,i])
    vpred = model.predict(X_val)
    print(f"{col} R2:", r2_score(y_val[:,i], vpred))
    preds[col] = model


submission = pd.DataFrame({"sample_id": test["sample_id"]})

for col,model in preds.items():
    submission[col] = model.fit(X_train, targets[col].values).predict(X_test)


sub_long = submission.melt(id_vars="sample_id",var_name="target_name",value_name="target")
sub_long.to_csv("submission.csv",index=False)

print("✅ submission.csv created")
print(submission.head())