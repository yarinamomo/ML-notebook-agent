# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# === BEFORE (original) ===
# import numpy as np
# 
# # Load GloVe embeddings
# def load_glove_embeddings(embeddings_file):
#     embeddings_index = dict()
#     with open(embeddings_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs
#     return embeddings_index
# 
# glove_embeddings_file = 'data/glove.6B.50d.txt'
# glove_embeddings = load_glove_embeddings(glove_embeddings_file)

# === AFTER (edited) ===
import numpy as np


def load_glove_embeddings(embeddings_file):
    embeddings_index = dict()
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings_file = 'data/glove.6B.50d.txt'
# glove_embeddings = load_glove_embeddings(glove_embeddings_file)

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
from transformers import BertTokenizer

# Load BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and preprocess text
def tokenize_text(text):
    tokens = bert_tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    return input_ids

# Example usage
text = "This is an example sentence."
input_ids = tokenize_text(text)


#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
docs = np.array(['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.'])
# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
# === BEFORE (original) ===
# import numpy as np
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# 
# max_length = 768
# 
# docs = np.array([np.array(pad_sequences(tokenize_text(i)), maxlen=max_length) for i in docs])

# === AFTER (edited) ===
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

max_length = 768

docs = np.array([pad_sequences([tokenize_text(i)], maxlen=max_length)[0] for i in docs])