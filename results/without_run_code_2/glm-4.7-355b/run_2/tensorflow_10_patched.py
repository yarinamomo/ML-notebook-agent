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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# for reproducing and fixing purposes, due to the cadec dataset not found
# df = pd.concat([df_psytar.iloc[:df_psytar.shape[0]+1], df_cadec])
df=df_psytar

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
df_1 = df[df['ADR']==1]
df_0 = df[df['ADR']==0]

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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
train_data = df["sentences"]
train_labels = df['ADR']

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
        return outputs

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
model_name = 'bert-base-uncased'
model = tf.keras.Sequential()
model.add(HuggingFaceLayer(model_name=model_name))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 15}
# === BEFORE (original) ===
# # Compile and train the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, train_labels, epochs=10)

# === AFTER (edited) ===
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare input data from tokenized data
train_input = {
    'input_ids': train_df['input_ids'].tolist(),
    'attention_mask': train_df['attention_mask'].tolist(),
    'token_type_ids': train_df['token_type_ids'].tolist()
}
train_labels_arr = train_df['label'].tolist()

valid_input = {
    'input_ids': valid_df['input_ids'].tolist(),
    'attention_mask': valid_df['attention_mask'].tolist(),
    'token_type_ids': valid_df['token_type_ids'].tolist()
}
valid_labels_arr = valid_df['label'].tolist()

model.fit(train_input, train_labels_arr, epochs=10, validation_data=(valid_input, valid_labels_arr))