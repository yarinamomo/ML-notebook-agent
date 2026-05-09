# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import STOPWORDS
from collections import defaultdict
import random
import re
import string

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
df.sample(5)

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
class_names =['Not a disaster', 'Disaster']

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# word_count
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# unique_word_count
df['unique_word_count'] = df['text'].apply(lambda x: len(set(str(x).split())))

# stop_word_count
df['stop_word_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

# url_count
df['url_count'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() if 'http' in w or 'https' in w]))

# mean_word_length
df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# char_count
df['char_count'] = df['text'].apply(lambda x: len(str(x)))

# punctuation_count
df['punctuation_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# hashtag_count
df['hashtag_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))

# mention_count
df['mention_count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
## Truncate some extreme values for better visuals ##
df['word_count'].loc[df['word_count']>60] = 60 #truncation for better visuals
df['char_count'].loc[df['char_count']>350] = 350 #truncation for better visuals
df['punctuation_count'].loc[df['punctuation_count']>10] = 10 #truncation for better visuals

f, axes = plt.subplots(3, 1, figsize=(20,30))
sns.boxplot(x='target', y='word_count', data=df, ax=axes[0])
axes[0].set_xlabel('Target', fontsize=12)
axes[0].set_title("Number of words in each class", fontsize=15)

sns.boxplot(x='target', y='char_count', data=df, ax=axes[1])
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_title("Number of characters in each class", fontsize=15)

sns.boxplot(x='target', y='punctuation_count', data=df, ax=axes[2])
axes[2].set_xlabel('Target', fontsize=12)
#plt.ylabel('Number of punctuations in text', fontsize=12)
axes[2].set_title("Number of punctuations in each class", fontsize=15)

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 6}
df['hashtag_count'].loc[df['hashtag_count']>60] = 60 #truncation for better visuals
df['mention_count'].loc[df['mention_count']>60] = 60 #truncation for better visuals

f, axes = plt.subplots(3, 1, figsize=(20,30))

sns.boxplot(x='target', y='hashtag_count', data=df, ax=axes[0])
axes[0].set_xlabel('Target', fontsize=12)
axes[0].set_title("Number of Hashtags in each class", fontsize=15)

sns.boxplot(x='target', y='mention_count', data=df, ax=axes[1])
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_title("Number of Mentions in each class", fontsize=15)

sns.boxplot(x='target', y='url_count', data=df, ax=axes[2])
axes[2].set_xlabel('Target', fontsize=12)
axes[2].set_title("Number of URLs in each class", fontsize=15)
plt.show()

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
exclude = string.punctuation
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

def remove_punc(text):
    return text.translate(str.maketrans('', '', exclude))

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 8}
text1 = 'Check out my notebook https://www.kaggle.com/campusx/notebook8223fc1abb'
text2 = '!hello *world@ 1'
df['text'] = df['text'].apply(remove_url)
#df['text'] = df['text'].apply(remove_punc)
test_df['text'] = test_df['text'].apply(remove_url)
#test_df['text'] = test_df['text'].apply(remove_punc)

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 9}
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 10}
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 11}
example_text = 'I will watch #Memento tonight!'
bert_input = tokenizer(example_text, padding='max_length', max_length = 15,
                      truncation = True, return_tensors = 'pt')

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 12}
input_ids = torch.tensor(bert_input.input_ids)
attention_mask = torch.tensor(bert_input.attention_mask)
bert_model = BertModel.from_pretrained('bert-base-uncased')
last_hidden_state, pooled_output = bert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict =False)

print(last_hidden_state.shape)
print(bert_model.config.hidden_size)

#%%
# --- [CELL 12]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 13}
# === BEFORE (original) ===
# import tensorflow as tf
# tf.keras.utils.plot_model(bert_model)

# === AFTER (edited) ===
import torch
from torchinfo import summary

summary(bert_model, input_data={"input_ids": input_ids, "attention_mask": attention_mask})