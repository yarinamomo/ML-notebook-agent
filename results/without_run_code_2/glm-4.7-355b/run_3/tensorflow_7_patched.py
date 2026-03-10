# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import tensorflow as tf
#import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizer, TFBertModel

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
try:
    tpu=tf.distribute.cluster_resolver.TCPClusterResolver()# this is a TensorFlow class
    #used to create a TPUStrategy object for training on TPUs.
    
    print("Device : ",tpu.master())#returns name of TPU device designated as master (the master manages training and resources)
    tf.config.experimental_connect_to_cluster(tpu)#connect TF class to cluster
    
    tf.tpu.experimental.initialize_tpu_system(tpu)# initializing TPU system
    
    strategy=tf.distribute.experimental.TPUStrategy(tpu)# distribute learning across multiple TPUs
    
except:
    strategy=tf.distribute.get_strategy()#returns current strategy 
    # for CPU and single GPU see https://www.kaggle.com/code/anasofiauzsoy/tutorial-notebook/notebook

print("Number of replicas : ",strategy.num_replicas_in_sync)# shows copies of model undergoing training that are used to
#synchronize gradient later


print(tf.__version__)

# Number of replicas :  1
# 2.6.4

#NOTE: NUmber of replicas is 1. It means 1 copy of the model is going through the process and is going to have the gradient sync.
#NOTE: The try block most proably did not work.

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
import pandas as pd

train = pd.read_csv("data/train.csv")
train = train[:8] # for faster reproducing and fixing purposes --- make a smaller dataset

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
def encode_sentence(s):
   tokens = list(tokenizer.tokenize(s))
   tokens.append('[SEP]')
   return tokenizer.convert_tokens_to_ids(tokens)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
def bert_encode(hypotheses, premises, tokenizer):
    
  num_examples = len(hypotheses)
  
  sentence1 = tf.ragged.constant([
      encode_sentence(s)
      for s in np.array(hypotheses)])
  sentence2 = tf.ragged.constant([
      encode_sentence(s)
       for s in np.array(premises)])

  cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
  input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

  input_mask = tf.ones_like(input_word_ids).to_tensor()

  type_cls = tf.zeros_like(cls)
  type_s1 = tf.zeros_like(sentence1)
  type_s2 = tf.ones_like(sentence2)
  input_type_ids = tf.concat(
      [type_cls, type_s1, type_s2], axis=-1).to_tensor()

  inputs = {
      'input_word_ids': input_word_ids.to_tensor(),
      'input_mask': input_mask,
      'input_type_ids': input_type_ids}

  return inputs

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
train_input = bert_encode(train.premise.values, train.hypothesis.values, tokenizer)

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
# === BEFORE (original) ===
# max_len = 50
# from transformers import BertTokenizer, TFBertModel
# 
# 
# def build_model():
#     bert_encoder = TFBertModel.from_pretrained(model_name)
#     input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
#     input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
#     input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")
#     
#     embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
#     output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])
#     
#     model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
#     model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     
#     return model

# === AFTER (edited) ===
max_len = 172
from transformers import BertTokenizer, TFBertModel


def build_model():
    bert_encoder = TFBertModel.from_pretrained(model_name)
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    input_type_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_type_ids")

    embedding = bert_encoder([input_word_ids, input_mask, input_type_ids])[0]
    output = tf.keras.layers.Dense(3, activation='softmax')(embedding[:,0,:])

    model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=output)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
with strategy.scope():
    model = build_model()
    model.summary()

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
model.fit(train_input, train.label.values, epochs = 2, verbose = 1, batch_size = 64, validation_split = 0.2)