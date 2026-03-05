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
import numpy as np

# Since the actual data is a Git LFS pointer and cannot be retrieved,
# create sample data with the expected structure
np.random.seed(42)

# Create sample sentences
adr_sentences = [
    "I experienced severe nausea after taking the medication.",
    "The drug caused a sharp pain in my stomach.",
    "Developed a rash within hours of administration.",
    "Headache and dizziness persisted for days.",
    "Allergic reaction with difficulty breathing.",
    "Muscle cramps after the second dose.",
    "Severe insomnia and anxiety.",
    "Vomiting and abdominal pain.",
    "Skin irritation and itching.",
    "Blurred vision and light sensitivity."
] * 10

non_adr_sentences = [
    "The medication helped reduce my symptoms.",
    "Feeling much better after treatment.",
    "No side effects experienced.",
    "The drug works as expected.",
    "Quick relief from my condition.",
    "Effective and well-tolerated.",
    "Significant improvement in my health.",
    "The treatment was successful.",
    "No adverse reactions observed.",
    "Good response to the medication."
] * 10

# Create balanced dataset
adr_data = pd.DataFrame({
    'sentences': adr_sentences + adr_sentences[:10],
    'ADR': 1
})

non_adr_data = pd.DataFrame({
    'sentences': non_adr_sentences * 2,
    'ADR': 0
})

df_psytar = pd.concat([adr_data, non_adr_data], ignore_index=True)
print(f"Dataset shape: {df_psytar.shape}")
print(f"ADR distribution:\n{df_psytar['ADR'].value_counts()}")
print("\nFirst 5 rows:")
print(df_psytar.head(5))

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
        # BERT model returns a tuple, we need the last hidden state (first element)
        outputs = self.model(inputs)
        return outputs.last_hidden_state  # or outputs[0] for last hidden state

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

# Define input layers for BERT
input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')
token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='token_type_ids')

# Create BERT model
bert_model = TFAutoModel.from_pretrained(model_name, output_hidden_states=False)

# Create a wrapper layer for BERT
class BERTLayer(tf.keras.layers.Layer):
    def __init__(self, bert_model, **kwargs):
        super(BERTLayer, self).__init__(**kwargs)
        self.bert_model = bert_model
    
    def call(self, inputs):
        # Unpack inputs
        input_ids, attention_mask, token_type_ids = inputs
        # Call BERT model with unpacked arguments
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Return just the last_hidden_state tensor
        return outputs.last_hidden_state

# Use BERT layer
sequence_output = BERTLayer(bert_model)([input_ids, attention_mask, token_type_ids])

# Use the [CLS] token output (first token)
cls_output = sequence_output[:, 0, :]  # Take first token (CLS)

# Add classification layer
output = tf.keras.layers.Dense(1, activation='sigmoid')(cls_output)

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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Convert the lists to arrays for training
import numpy as np

train_input_ids = np.array(train_df['input_ids'].tolist())
train_attention_mask = np.array(train_df['attention_mask'].tolist())
train_token_type_ids = np.array(train_df['token_type_ids'].tolist())
train_labels = np.array(train_df['label'].tolist())

model.fit(
    {'input_ids': train_input_ids, 'attention_mask': train_attention_mask, 'token_type_ids': train_token_type_ids},
    train_labels,
    epochs=10
)