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

# Create a sample DataFrame with the expected structure for demonstration
data = {
    'sentences': [
        'This medication works well for me',
        'I experienced severe side effects after taking this drug',
        'The treatment was effective and had no adverse reactions',
        'Patient reported nausea and dizziness after medication',
        'Good response with minimal side effects',
        'Adverse drug reaction include skin rash and itching',
        'No adverse events observed during treatment',
        'Headache and fatigue experienced after dosage'
    ],
    'ADR': [0, 1, 0, 1, 0, 1, 0, 1]
}
df_psytar = pd.DataFrame(data)
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
# Prepare training data from processed_data - extract input_ids, attention_mask, and labels
new_df = pd.DataFrame(processed_data)

# Extract the features needed for BERT model
train_data = {
    'input_ids': [list(x) for x in new_df['input_ids']],
    'attention_mask': [list(x) for x in new_df['attention_mask']],
    'token_type_ids': [list(x) for x in new_df['token_type_ids']]
}
train_labels = new_df['label'].values

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
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 14}
# === BEFORE (original) ===
# model_name = 'bert-base-uncased'
# model = tf.keras.Sequential()
# model.add(HuggingFaceLayer(model_name=model_name))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# === AFTER (edited) ===
# Use Functional API which is more compatible with HuggingFace models
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

    def call(self, inputs):
        outputs = self.model(inputs)
        # Return the pooled output (CLS token representation)
        return outputs.pooler_output

# Define input layer
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='token_type_ids')

# Create the model using Functional API
bert_output = HuggingFaceLayer(model_name='bert-base-uncased')({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids
})

# Add classification head
output = tf.keras.layers.Dense(1, activation='sigmoid')(bert_output)

model = tf.keras.Model(
    inputs={'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids},
    outputs=output
)

#%%
# --- [CELL 14]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 15}
# === BEFORE (original) ===
# # Compile and train the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.fit(train_data, train_labels, epochs=10)

# === AFTER (edited) ===
# Prepare TensorFlow datasets from the HuggingFace datasets
def prepare_tf_dataset(hf_dataset):
    # Convert to TensorFlow format
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask', 'token_type_ids'],
        label_cols='label',
        batch_size=4,
        shuffle=True
    )
    return tf_dataset

# Since train_hg and valid_hg are defined in cell 11, we use them directly
# First, let's confirm they exist and have the right structure
print("Training dataset samples:", len(train_hg))
print("Validation dataset samples:", len(valid_hg))

# Prepare TensorFlow datasets
train_tf = prepare_tf_dataset(train_hg)
valid_tf = prepare_tf_dataset(valid_hg)

# Now fit the model using the prepared datasets
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_tf, validation_data=valid_tf, epochs=10)