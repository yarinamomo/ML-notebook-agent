# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
import logging  # 日志相关的包
import random

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# from gensim.models.word2vec import Word2Vec
# 
# num_features = 10 #100  # 词向量维度
# num_workers = 8
# 
# # train_df = pd.read_csv('data/train_set.csv.zip', sep='\t')
# train_df = pd.read_csv('data/train_set.csv.zip', sep='\t', nrows=5000)
# train_text = list(map(lambda x:list(x.split()), train_df.iloc[:, 1]))
# model = Word2Vec(train_text, workers=num_workers, vector_size=num_features)
# model.init_sims(replace=True)
# 
# model.wv.save_word2vec_format('data/word2vec.txt', binary=False)

# === AFTER (edited) ===
from gensim.models.word2vec import Word2Vec

num_features = 10
num_workers = 8

# Create synthetic training data since the zip file is not available
# Simulating text classification data with text and label columns
texts = [
    '这是一篇关于科技发展的重要文章',
    '股票市场今天出现了大幅波动',
    '体育比赛非常精彩令人难忘',
    '娱乐新闻吸引了众多观众',
    '时政报道反映了当前社会状况',
    '社会发展需要更多关注',
    '教育改革是时代发展的必然',
    '财经新闻关注经济发展趋势',
    '家居设计体现了生活品质',
    '游戏产业增长迅速令人瞩目',
    '房产市场始终保持稳定增长',
    '时尚潮流引领着消费观念',
    '彩票中奖需要一定的运气',
    '星座运势受到许多人关注',
]

# Generate more samples by random combinations
np.random.seed(2023)
additional_texts = []
for i in range(5000 - len(texts)):
    t1 = np.random.choice(texts)
    t2 = np.random.choice(texts[7:])  # Use shorter sentences
    additional_texts.append(t1 + ' ' + t2)

all_texts = texts + additional_texts
labels = [i % 14 for i in range(len(all_texts))]

train_df = pd.DataFrame({'text': all_texts, 'label': labels})

# Train Word2Vec on the text data
train_text = list(map(lambda x: list(x.split()), train_df['text']))
model = Word2Vec(train_text, workers=num_workers, vector_size=num_features)
model.init_sims(replace=True)

# Save the word2vec embeddings
import os
os.makedirs('data', exist_ok=True)
model.wv.save_word2vec_format('data/word2vec.txt', binary=False)

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
from collections import Counter
from transformers import BasicTokenizer

basic_tokenizer = BasicTokenizer()

class Vocab():
    def __init__(self, train_data):
        self.min_count = 5
        self.pad = 0
        self.unk = 1
        self._id2word = ['[PAD]', '[UNK]']
        self._id2extword = ['[PAD]', '[UNK]']
        
        self._id2label = []
        self.target_names = []
        
        self.build_vocab(train_data)
        
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._label2id = reverse(self._id2label)
        
    def build_vocab(self, data):
        self.word_counter = Counter()
        
        for text in data['text']:
            words = text.split()
            for word in words:
                self.word_counter[word] += 1
                
        for word, count in self.word_counter.most_common():
            if count >= self.min_count:
                self._id2word.append(word)
                
        label2name = {0: '科技', 1: '股票', 2: '体育', 3: '娱乐', 4: '时政', 5: '社会', 6: '教育',
                      7: '财经', 8: '家居', 9: '游戏', 10: '房产', 11: '时尚', 12: '彩票', 13: '星座'}
        self.label_counter = Counter(data['label'])
        
        for label in range(len(self.label_counter)):
            count = self.label_counter[label]
            self._id2label.append(label)
            self.target_names.append(label2name[label])
            
    def load_pretrained_embs(self, embfile):
        with open(embfile, encoding='utf-8') as f:
            lines = f.readlines()
            items = lines[0].split()
            word_count, embedding_dim = int(items[0]), int(items[1])
            
        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        for line in lines[1:]:
            values = line.split()
            self._id2extword.append(values[0])
            vector = np.array(values[1:], dtype='float64')
            embeddings[self.unk] += vector
            embeddings[index] = vector
            index += 1
            
        embeddings[self.unk] = embeddings[self.unk] / word_count
        embeddings = embeddings / np.std(embeddings)
        
        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)
        
        assert len(set(self._id2extword)) == len(self._id2extword)
        
        return embeddings
    
    def word2id(self, xs):
        if isinstance(xs, list): 
            return [self._word2id.get(x, self.unk) for x in xs]
        return self._word2id.get(xs, self.unk)
    
    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.unk) for x in xs]
        return self._extword2id.get(xs, self.unk)
    
    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.unk) for x in xs]
        return self._label2id.get(xs, self.unk)
    
    def word_size(self):
        return len(self._id2word)
    
    def extword_size(self):
        return len(self._id2extword)
    
    def label_size(self):
        return len(self._id2label)
    
vocab = Vocab(train_df)
            

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 5}
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    '''Scaled Dot-Product Attention'''
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)
        
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))
        
        self.query = nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)
        
    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: batch_size x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks: batch_size x len
        
        # broadcast机制
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden
        
        outputs = torch.matmul(key, self.query)  # b x len
        
        # 填充一个很小的负数，softmax后就会变为0
        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32)) 
        
        attn_scores = F.softmax(masked_outputs, dim=1)  # b x len
        
        # 经过softmax后可能存在nan，因此将这些位置都变为0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)
        
        # 矩阵批量乘法（batch matrix-matrix product）函数
        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden
        
        return batch_outputs, attn_scores
    

#%%
# --- [CELL 5]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# word2vec_path = 'data/word2vec.txt'
# dropout = 0.15
# word_hidden_size = 128
# word_num_layers = 2
# 
# class WordLSTMEncoder(nn.Module):
#     '''
#     结合word2vec（预训练）和nn.Embedding()（待训练）的词向量表示，然后进一步用lstm提取序列信息，更新词向量表示
#     '''
#     def __init__(self, vocab):
#         super(WordLSTMEncoder, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         self.word_dims = num_features #100
#         
#         self.word_embed = nn.Embedding(vocab.word_size, self.word_dims, padding_idx=0)
#         
#         extword_embed = vocab.load_pretrained_embs(word2vec_path)
#         extword_size, word_dims = extword_embed.shape
#         
#         self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
#         self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
#         self.extword_embed.weight.requires_grad = False
#         
#         input_size = self.word_dims
#         
#         self.word_lstm = nn.LSTM(input_size=input_size, 
#                                  hidden_size=word_hidden_size,
#                                  num_layers=word_num_layers,  # LSTM层的数量
#                                  batch_first=True,
#                                  bidirectional=True)
#         
#     def forward(self, word_ids, extword_ids, batch_masks):
#         # word_ids: sen_num x sent_len
#         # extword_ids: sen_num x sent_len
#         # batch_masks: sen_num x sent_len
#         
#         word_embed = self.word_embed(word_ids)  # sen_num x sent_len x 100
#         extword_embed = self.extword_embed(extword_ids)
#         batch_embed = word_embed + extword_embed
#         
#         if self.training:
#             batch_embed = self.dropout(batch_embed)
#             
#         hiddens, _ = self.word_lstm(batch_embed)  # sen_num x sent_len x hidden*2
#         hiddens = hiddens * batch_masks.unsqueeze(2)
#         
#         if self.training:
#             hiddens = self.dropout(hiddens)
#             
#         return hiddens

# === AFTER (edited) ===
word2vec_path = 'data/word2vec.txt'
dropout = 0.15
word_hidden_size = 128
word_num_layers = 2

class WordLSTMEncoder(nn.Module):
    '''
    结合word2vec（预训练）和nn.Embedding()（待训练）的词向量表示，然后进一步用lstm提取序列信息，更新词向量表示
    '''
    def __init__(self, vocab):
        super(WordLSTMEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.word_dims = num_features

        self.word_embed = nn.Embedding(vocab.word_size(), self.word_dims, padding_idx=0)

        extword_embed = vocab.load_pretrained_embs(word2vec_path)
        extword_size, word_dims = extword_embed.shape

        self.extword_embed = nn.Embedding(extword_size, word_dims, padding_idx=0)
        self.extword_embed.weight.data.copy_(torch.from_numpy(extword_embed))
        self.extword_embed.weight.requires_grad = False

        input_size = self.word_dims

        self.word_lstm = nn.LSTM(input_size=input_size,
                                 hidden_size=word_hidden_size,
                                 num_layers=word_num_layers,
                                 batch_first=True,
                                 bidirectional=True)

    def forward(self, word_ids, extword_ids, batch_masks):




        word_embed = self.word_embed(word_ids)
        extword_embed = self.extword_embed(extword_ids)
        batch_embed = word_embed + extword_embed

        if self.training:
            batch_embed = self.dropout(batch_embed)

        hiddens, _ = self.word_lstm(batch_embed)
        hiddens = hiddens * batch_masks.unsqueeze(2)

        if self.training:
            hiddens = self.dropout(hiddens)

        return hiddens

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 7}
sent_hidden_size = 256
sent_num_layers = 2

class SentEncoder(nn.Module):
    '''句子级别的语义表示'''
    def __init__(self, sent_rep_size):
        super(SentEncoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.sent_lstm = nn.LSTM(input_size=sent_rep_size, 
                                 hidden_size=sent_hidden_size,
                                 num_layers=sent_num_layers,
                                 batch_first=True,
                                 bidirectional=True)
        
    def forward(self, sent_reps, sent_masks):
        
        sent_hiddens, _ = self.sent_lstm(sent_reps)
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)
        
        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)
            
        return sent_hiddens

#%%
# --- [CELL 7]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# class Model(nn.Module):
#     def __init__(self, vocab):
#         super(Model, self).__init__()
#         self.sent_rep_size = word_hidden_size * 2  # 双向lstm，每个词向量对应的隐藏层维度
#         self.doc_rep_size = sent_hidden_size * 2  # 文档表示大小，每个句子对应隐藏层的维度
#         self.all_parameters = {}
#         
#         parameters = []
#         self.word_encoder = WordLSTMEncoder(vocab)
#         self.word_attention = Attention(self.sent_rep_size)
#         # filter(判断函数, 可迭代对象)
#         parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
#         parameters.extend(list(filter(lambda p: p.requires_grad, self.word_attention.parameters())))
#         self.sent_encoder = SentEncoder(self.sent_rep_size)
#         self.sent_attention = Attention(self.doc_rep_size)
#         parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
#         parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
#         self.out = nn.Linear(self.doc_rep_size, vocab.label_size, bias=True)
#         parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))
#         
#         self.to(device)
#         
#         if len(parameters) > 0:
#             self.all_parameters['basic_parameters'] = parameters
#             
#         # 模型总参数量
#         para_num = sum([np.prod(list(p.size())) for p in self.parameters()])
#         
#     def forward(self, batch_inputs):
#         # batch_inputs(batch_inputs1, batch_inputs2): b x doc_len x sent_len
#         # batch_masks: b x doc_len x sent_len
#         # 不明白为什么这里要同时输入两个batch？？？？？？？
#         batch_inputs1, batch_inputs2, batch_masks = batch_inputs
#         batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
#         batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)
#         batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)
#         batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)
#         batch_hiddens = self.word_encoder(batch_inputs1, batch_inputs2, batch_masks)
#         sent_reps, atten_scores = self.word_attention(batch_hiddens, batch_masks)
#         sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)
#         batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)
#         sent_masks = batch_masks.bool().any(2).float()
#         sent_hiddens = self.sent_encoder(sent_reps, sent_masks)
#         doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)
#         batch_outputs = self.out(doc_reps)
#         
#         return batch_outputs
#     
# model = Model(vocab)

# === AFTER (edited) ===
class Model(nn.Module):
    def __init__(self, vocab):
        super(Model, self).__init__()
        self.sent_rep_size = word_hidden_size * 2
        self.doc_rep_size = sent_hidden_size * 2
        self.all_parameters = {}

        parameters = []
        self.word_encoder = WordLSTMEncoder(vocab)
        self.word_attention = Attention(self.sent_rep_size)

        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.word_attention.parameters())))
        self.sent_encoder = SentEncoder(self.sent_rep_size)
        self.sent_attention = Attention(self.doc_rep_size)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_encoder.parameters())))
        parameters.extend(list(filter(lambda p: p.requires_grad, self.sent_attention.parameters())))
        self.out = nn.Linear(self.doc_rep_size, vocab.label_size(), bias=True)
        parameters.extend(list(filter(lambda p: p.requires_grad, self.out.parameters())))

        self.to(device)

        if len(parameters) > 0:
            self.all_parameters['basic_parameters'] = parameters


        para_num = sum([np.prod(list(p.size())) for p in self.parameters()])

    def forward(self, batch_inputs):



        batch_inputs1, batch_inputs2, batch_masks = batch_inputs
        batch_size, max_doc_len, max_sent_len = batch_inputs1.shape[0], batch_inputs1.shape[1], batch_inputs1.shape[2]
        batch_inputs1 = batch_inputs1.view(batch_size * max_doc_len, max_sent_len)
        batch_inputs2 = batch_inputs2.view(batch_size * max_doc_len, max_sent_len)
        batch_masks = batch_masks.view(batch_size * max_doc_len, max_sent_len)
        batch_hiddens = self.word_encoder(batch_inputs1, batch_inputs2, batch_masks)
        sent_reps, atten_scores = self.word_attention(batch_hiddens, batch_masks)
        sent_reps = sent_reps.view(batch_size, max_doc_len, self.sent_rep_size)
        batch_masks = batch_masks.view(batch_size, max_doc_len, max_sent_len)
        sent_masks = batch_masks.bool().any(2).float()
        sent_hiddens = self.sent_encoder(sent_reps, sent_masks)
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)
        batch_outputs = self.out(doc_reps)

        return batch_outputs

model = Model(vocab)