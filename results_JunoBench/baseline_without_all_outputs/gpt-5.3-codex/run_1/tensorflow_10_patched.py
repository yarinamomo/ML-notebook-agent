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
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
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

    def call(self, inputs, **kwargs):
        outputs = self.model(inputs, **kwargs)
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
# Prepare tokenized tensors for Keras training
train_enc = tokenizer(
    train_data.tolist(),
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="tf"
)

# Build a model that accepts BERT inputs and outputs a binary prediction
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

bert_outputs = HuggingFaceLayer(model_name=model_name)(
    {"input_ids": input_ids, "attention_mask": attention_mask}
)

# Use pooled output when available, otherwise CLS token from last_hidden_state
if hasattr(bert_outputs, "pooler_output") and bert_outputs.pooler_output is not None:
    x = bert_outputs.pooler_output
else:
    x = bert_outputs.last_hidden_state[:, 0, :]

output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(
    {"input_ids": train_enc["input_ids"], "attention_mask": train_enc["attention_mask"]},
    tf.convert_to_tensor(train_labels.values, dtype=tf.float32),
    epochs=10,
    batch_size=16
)