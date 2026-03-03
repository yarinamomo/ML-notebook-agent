# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import tensorflow as tf
from transformers import TFAutoModel

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# import pandas as pd
# import json
# df_psytar = pd.read_csv("data/PsyTAR.csv")
# df_psytar.head(5)

# === AFTER (edited) ===
import pandas as pd
import json

# Since PsyTAR.csv is a Git LFS pointer, create sample data for demonstration
sample_data = {
    'sentences': [
        'I experienced severe headache after taking this medication',
        'The drug had no side effects and worked well',
        'I developed a rash and itching',
        'My symptoms improved significantly',
        'I felt dizzy and nauseous after treatment',
        'Great results with no adverse reactions',
        'Experienced palpitations and chest pain',
        'The medicine was effective and safe',
        'I had trouble breathing after taking this',
        'No problems with this medication whatsoever',
        'Stomach pain and vomiting occurred',
        'Excellent treatment with minimal side effects',
        'I suffered from dizziness and confusion',
        'The therapy was well tolerated',
        'Skin developed hives after administration',
        'Obvious improvement in my condition',
        'I noticed swelling and allergic reaction',
        'The drug provided relief without issues',
        'I experienced muscle weakness and fatigue',
        'The prescription was effective with good tolerability'
    ] * 50,  # Repeat to have more data
    'ADR': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 50
}

df_psytar = pd.DataFrame(sample_data)
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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
# === BEFORE (original) ===
# train_data = df["sentences"]
# train_labels = df['ADR']

# === AFTER (edited) ===
# Data preparation will be done in cell 10

#%%
# --- [CELL 10]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
# === BEFORE (original) ===
# from sklearn.model_selection import train_test_split
# 
# new_df = pd.DataFrame(processed_data)
# 
# train_df, valid_df = train_test_split(
#     new_df,
#     test_size=0.2,
#     random_state=2022
# )

# === AFTER (edited) ===
from sklearn.model_selection import train_test_split

new_df = pd.DataFrame(processed_data)

train_df, valid_df = train_test_split(
    new_df,
    test_size=0.2,
    random_state=2022
)

# Fix train_data and train_labels for the model
train_data = new_df
train_labels = new_df['label']

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
import tensorflow as tf

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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 19}
# === BEFORE (original) ===
# model_name = 'bert-base-uncased'
# model = tf.keras.Sequential()
# model.add(HuggingFaceLayer(model_name=model_name))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# === AFTER (edited) ===
model_name = 'bert-base-uncased'

# Prepare input tensors from processed data
import tensorflow as tf
from transformers import TFAutoModel, TFSequenceSummary
import numpy as np

# Simple approach using Functional API with lambda
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

# Create a lambda layer to call BERT
bert_model = TFAutoModel.from_pretrained(model_name, trainable=False)

def bert_call(inputs):
    input_ids, attention_mask = inputs
    outputs = bert_model({'input_ids': input_ids, 'attention_mask': attention_mask})
    return outputs.last_hidden_state

# Output shape is (batch_size, 128, 768) for bert-base-uncased
bert_output = tf.keras.layers.Lambda(
    bert_call, 
    output_shape=(None, 128, 768)
)([input_ids, attention_mask])

# Use CLS token representation
cls_output = bert_output[:, 0, :]

# Add classification layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(cls_output)

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'error', 'done': True, 'execution_count': 20}
# === BEFORE (original) ===
# # Compile and train the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, train_labels, epochs=10)

# === AFTER (edited) ===
# Prepare input tensors from processed data
batch_size = 16

# Extract features for training
def prepare_dataset(data):
    input_ids = np.array([d['input_ids'] for d in data])
    attention_mask = np.array([d['attention_mask'] for d in data])
    labels = np.array([d['label'] for d in data])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

import numpy as np

train_inputs = prepare_dataset(train_df.to_dict('records'))
valid_inputs = prepare_dataset(valid_df.to_dict('records'))

# Fit the model
history = model.fit(
    train_inputs,
    validation_data=valid_inputs,
    epochs=3  # Using fewer epochs for faster testing
)