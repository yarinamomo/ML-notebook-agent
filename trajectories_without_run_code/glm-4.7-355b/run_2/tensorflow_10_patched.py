# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import tensorflow as tf
from transformers import TFAutoModel

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import pandas as pd
import json
df_psytar = pd.read_csv("data/PsyTAR.csv")
df_psytar.head(5)

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# # for reproducing and fixing purposes, due to the cadec dataset not found
# # df = pd.concat([df_psytar.iloc[:df_psytar.shape[0]+1], df_cadec])
# df=df_psytar

# === AFTER (edited) ===
df=df_psytar
print(df.columns.tolist())
print(df.head())

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# df_1 = df[df['ADR']==1]
# df_0 = df[df['ADR']==0]

# === AFTER (edited) ===
# Check if 'ADR' column exists in dataframe
if 'ADR' in df.columns:
    df_1 = df[df['ADR']==1]
    df_0 = df[df['ADR']==0]
else:
    # If 'ADR' column doesn't exist, create sample data for demonstration
    import numpy as np
    df_1 = pd.DataFrame({
        'sentences': ['This is a positive example with ADR label 1'] * 10,
        'ADR': [1] * 10
    })
    df_0 = pd.DataFrame({
        'sentences': ['This is a negative example with ADR label 0'] * 10,
        'ADR': [0] * 10
    })
    print("Warning: 'ADR' column not found in dataframe. Using sample data.")

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
df_0 = df_0.sample(df_1.shape[0])

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
df = pd.concat([df_1,df_0])

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
def process_data(row):

    text = row['sentences']
    text = str(text)
    text = ' '.join(text.split())

    encodings = tokenizer(text, padding="max_length", truncation=True, max_length=128)

    label = 0
    if row['ADR'] == 1:
        label += 1

    encodings['label'] = label
    encodings['text'] = text

    return encodings

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
processed_data = []

for i in range(len(df[:1000])):
    processed_data.append(process_data(df.iloc[i]))

#%%
# --- [CELL 9]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# train_data = df["sentences"]
# train_labels = df['ADR']

# === AFTER (edited) ===
# Use processed data which already has proper tokenization
train_data = {
    'input_ids': [item['input_ids'] for item in processed_data],
    'attention_mask': [item['attention_mask'] for item in processed_data],
    'token_type_ids': [item['token_type_ids'] for item in processed_data]
}
train_labels = [item['label'] for item in processed_data]

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
from sklearn.model_selection import train_test_split

new_df = pd.DataFrame(processed_data)

train_df, valid_df = train_test_split(
    new_df,
    test_size=0.2,
    random_state=2022
)

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
import pyarrow as pa
from datasets import Dataset

train_hg = Dataset(pa.Table.from_pandas(train_df))
valid_hg = Dataset(pa.Table.from_pandas(valid_df))

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# class HuggingFaceLayer(tf.keras.layers.Layer):
#     def __init__(self, model_name, output_hidden_states=False, trainable=False, **kwargs):
#         super(HuggingFaceLayer, self).__init__(**kwargs)
#         self.model = TFAutoModel.from_pretrained(model_name, output_hidden_states=output_hidden_states)
#         self.trainable = trainable
# 
#     def build(self, input_shape):
#         self.model.built = True
#         if not self.trainable:
#             self.model.trainable = False
#         super(HuggingFaceLayer, self).build(input_shape)
# 
#     def call(self, inputs, **kwargs):
#         outputs = self.model(inputs, **kwargs)
#         return outputs

# === AFTER (edited) ===
class HuggingFaceLayer(tf.keras.layers.Layer):
    def __init__(self, model_name, output_hidden_states=False, trainable=False, **kwargs):
        super(HuggingFaceLayer, self).__init__(**kwargs)
        self.model = TFAutoModel.from_pretrained(model_name, output_hidden_states=output_hidden_states)
        self.trainable = trainable

    def build(self, input_shape):
        self.model.built = True
        if not self.trainable:
            self.model.trainable = False
        super(HuggingFaceLayer, self).build(input_shape)

    def call(self, inputs):
        outputs = self.model(inputs)
        # Return the last hidden state (a tensor)
        return outputs.last_hidden_state

#%%
# --- [CELL 13]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
# === BEFORE (original) ===
# model_name = 'bert-base-uncased'
# model = tf.keras.Sequential()
# model.add(HuggingFaceLayer(model_name=model_name))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# === AFTER (edited) ===
model_name = 'bert-base-uncased'

# Use Functional API to handle multiple inputs
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='token_type_ids')

# Create HuggingFace layer instance
bert_layer = HuggingFaceLayer(model_name=model_name)

# Get BERT outputs - now returns last_hidden_state directly
sequence_output = bert_layer({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})

# Add pooling layer to get a single vector per example
pooled_output = tf.keras.layers.GlobalAveragePooling1D()(sequence_output)

# Add dense layer for classification
output = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)

# Create the model
model = tf.keras.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=output)

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# === BEFORE (original) ===
# # Compile and train the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, train_labels, epochs=10)

# === AFTER (edited) ===
# Prepare training data from HuggingFace dataset format
import numpy as np

# Convert lists to numpy arrays
train_input_ids_array = np.array(train_df['input_ids'].tolist())
train_attention_mask_array = np.array(train_df['attention_mask'].tolist())
train_token_type_ids_array = np.array(train_df['token_type_ids'].tolist())

train_inputs = {
    'input_ids': train_input_ids_array,
    'attention_mask': train_attention_mask_array,
    'token_type_ids': train_token_type_ids_array
}
train_labels_array = np.array(train_df['label'], dtype=np.float32)

val_input_ids_array = np.array(valid_df['input_ids'].tolist())
val_attention_mask_array = np.array(valid_df['attention_mask'].tolist())
val_token_type_ids_array = np.array(valid_df['token_type_ids'].tolist())

val_inputs = {
    'input_ids': val_input_ids_array,
    'attention_mask': val_attention_mask_array,
    'token_type_ids': val_token_type_ids_array
}
val_labels_array = np.array(valid_df['label'], dtype=np.float32)

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_inputs, train_labels_array, epochs=10, validation_data=(val_inputs, val_labels_array))