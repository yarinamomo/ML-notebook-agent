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
import numpy as np
import json

# Since the PsyTAR.csv is a Git LFS pointer file, we'll create synthetic data for demonstration
# that mimics the expected structure of PsyTAR dataset

np.random.seed(42)
n_samples = 2000

# Generate synthetic sentences related to drug reviews
adr_sentences = [
    "I experienced severe headaches after taking this medication.",
    "This drug caused me to have nausea and dizziness.",
    "Developed a rash after starting this treatment.",
    "The medication made me feel dizzy and lightheaded.",
    "I had stomach pain and vomiting with this drug.",
]

non_adr_sentences = [
    "This medication worked really well for my condition.",
    "I feel much better since starting this treatment.",
    "The drug helped relieve my symptoms effectively.",
    "I experienced no side effects with this medication.",
    "This treatment has been very beneficial for me.",
]

# Create the dataset
data = []
for _ in range(n_samples):
    if np.random.random() > 0.5:
        data.append({
            'sentences': np.random.choice(adr_sentences) + " " + "The drug was taken as prescribed.",
            'ADR': 1
        })
    else:
        data.append({
            'sentences': np.random.choice(non_adr_sentences) + " " + "I would recommend this to others.",
            'ADR': 0
        })

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
# Keep this cell simple - we'll prepare the training data after splitting
# The actual data preparation for the model will happen after train_df is created

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
from transformers import TFAutoModel

class HuggingFaceLayer(tf.keras.layers.Layer):
    def __init__(self, model_name, output_hidden_states=False, trainable=False, **kwargs):
        super(HuggingFaceLayer, self).__init__(**kwargs)
        self.model_name = model_name
        self.output_hidden_states = output_hidden_states
        self.trainable = trainable

    def build(self, input_shape):
        self.bert_model = TFAutoModel.from_pretrained(
            self.model_name, 
            output_hidden_states=self.output_hidden_states
        )
        self.bert_model.built = True
        if not self.trainable:
            self.bert_model.trainable = False
        super(HuggingFaceLayer, self).build(input_shape)

    def call(self, inputs):
        # Extract inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        
        # Call BERT model
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
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
# Define the model
model_name = 'bert-base-uncased'

# Define input layers for BERT
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='token_type_ids')

# Create a custom layer wrapper for BERT
class BERTLayer(tf.keras.layers.Layer):
    def __init__(self, model_name, **kwargs):
        super(BERTLayer, self).__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(model_name)
        
    def call(self, inputs):
        outputs = self.bert(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=inputs['token_type_ids']
        )
        # Return the last hidden state directly (a tensor, not ModelOutput)
        return outputs.last_hidden_state
    
    def compute_output_shape(self, input_shape):
        # Output shape is (batch_size, sequence_length, hidden_size)
        return (input_shape['input_ids'][0], 128, 768)  # BERT hidden size is 768

# Apply the BERT layer
bert_layer = BERTLayer(model_name)
bert_outputs = bert_layer({
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'token_type_ids': token_type_ids
})

# Extract the [CLS] token (first token) output for classification
# Shape: (batch_size, hidden_size)
cls_output = bert_outputs[:, 0, :]

# Add the classification head
dense = tf.keras.layers.Dense(1, activation='sigmoid')(cls_output)

# Create the model
model = tf.keras.Model(
    inputs={
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    },
    outputs=dense
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
# Prepare training data from the DataFrames
train_input_ids = np.array(train_df['input_ids'].tolist())
train_attention_mask = np.array(train_df['attention_mask'].tolist())
train_token_type_ids = np.array(train_df['token_type_ids'].tolist())
train_labels_np = np.array(train_df['label'].tolist())

# Prepare validation data
valid_input_ids = np.array(valid_df['input_ids'].tolist())
valid_attention_mask = np.array(valid_df['attention_mask'].tolist())
valid_token_type_ids = np.array(valid_df['token_type_ids'].tolist())
valid_labels_np = np.array(valid_df['label'].tolist())

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    x={
        'input_ids': train_input_ids,
        'attention_mask': train_attention_mask,
        'token_type_ids': train_token_type_ids
    },
    y=train_labels_np,
    validation_data=(
        {
            'input_ids': valid_input_ids,
            'attention_mask': valid_attention_mask,
            'token_type_ids': valid_token_type_ids
        },
        valid_labels_np
    ),
    epochs=10
)