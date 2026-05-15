# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
from datasets import load_dataset
import torch
import torch.nn.functional as F
import numpy as np
import time 
import pickle

dataset = load_dataset("sst", "default")
dataset2 = load_dataset("multi_nli")

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
torch.manual_seed = 555
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
# !wget http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove*.zip

glv = dict()
glv_size = 50
with open('data/glove.6B.{}d.txt'.format(glv_size),'r') as fp:
    for line in fp:
        word, *vec = line.split()
        glv[word] = torch.tensor(list(map(float , vec)))

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
embed = torch.zeros((len(glv)+2 , glv_size))
ind =2

word2index ={'_pad_':0}
index2word ={0:'_pad_'}

for x in glv:
    if(len(glv[x]) != glv_size):
        continue
    embed[ind] = glv[x]
    word2index[x] = ind
    index2word[ind] = x
    ind+=1

#%%
# --- [CELL 4]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
sentences = dataset['train']['sentence']
testsent = dataset['test']['sentence']

#%%
# --- [CELL 5]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))

def tokenize(sentences):
    tokens =[]
    max_len = 0
    for sentence in sentences:
        sentence = sentence.replace('\\',' ')
        sentence = sentence.replace('/',' ')
        sentence = sentence.replace('\'',' ')
        word_tokens = word_tokenize(sentence)
        max_len = max(max_len ,len( word_tokens))
        tokens.append([w for w in word_tokens if not w.lower() in stop_words and len(w)>2])
    return tokens , max_len

#%%
# --- [CELL 6]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
def build_vocab(sentences):
    vocab = set()
    X = list()
    y = list()
    
    for tokens in sentences:
        for token in tokens:
            vocab.add(token)
    vocab = list(vocab)
    int2text = dict()
    text2int = dict()
    vocab = ["_pad_"] + vocab
    for ind ,x in enumerate(vocab):
        int2text[ind] = x 
        text2int[x] = ind 
    
    return vocab ,int2text , text2int

#%%
# --- [CELL 7]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
def build_input(sentences ,word2index, text2int):
    X =[]
    Y =[]
    for tokens in sentences:
        curx = [word2index['_pad_']]
        cury = list()
        for token in tokens:
            if token in word2index:
                curx.append(word2index[token])
                cury.append(text2int[token])
            else:
                curx.append(1)
                cury.append(1)
        cury.append(text2int['_pad_'])
        X.append(torch.tensor(curx))
        Y.append(torch.tensor(cury))
    return X,Y

#%%
# --- [CELL 8]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
def build_input_test(sentences ,word2index):
    X =[]
    for tokens in sentences:
        curx = [word2index['_pad_']]
        for token in tokens:
            if token in word2index:
                curx.append(word2index[token])
            else:
                curx.append(1)
        X.append(torch.tensor(curx))
    return X

#%%
# --- [CELL 9]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
tokens , max_len = tokenize(sentences)
vocab , int2text , text2int = build_vocab(tokens)
X,Y = build_input(tokens , word2index,text2int)

#%%
# --- [CELL 10]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
from torch.utils.data import Dataset, DataLoader
class  data(Dataset):
    def __init__(self , X,Y,vs , padsz):
        self.X = X
        self.Y = Y
        self.vocab_size = vs
        self.mx = padsz
    def __len__(self):
        return len(self.X)
    def __getitem__(self , index):
        dif = len(self.mx - self.X[index] )
        _x = self.X[index]
        _y = self.Y[index]
        if dif > 0:
            a = torch.zeros(self.mx)
            b = torch.zeros(self.mx)
            a[:len(_x)] = _x
            b[:len(_y)] = _y
            _x = a
            _y = torch.zeros( ( self.mx, self.vocab_size))
            _y [torch.arange(self.mx),b.long()] =1
        return _x.long() , _y.long()

#%%
# --- [CELL 11]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
elmotrain = data(X,Y, len(vocab),40)

#%%
# --- [CELL 12]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
traindata = DataLoader(elmotrain, batch_size=32 )

#%%
# --- [CELL 13]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
tokens_test,ml = tokenize(testsent)
X_test = build_input_test(tokens_test , word2index)

#%%
# --- [CELL 14]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
class elmo(torch.nn.Module):
    def __init__(self , vocab_size,dim ,classes = 2, embed = None):
        super(elmo, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size , dim )
        if(embed != None):
          self.embedding.weights = torch.nn.Parameter(embed)
#         self.forward_lstm1 = torch.nn.LSTM(dim, dim)
#         self.forward_lstm2 = torch.nn.LSTM(dim, dim)
#         self.backward_lstm1 = torch.nn.LSTM(dim, dim)
#         self.backward_lstm2 = torch.nn.LSTM(dim, dim)
        self.bilstm1 = torch.nn.LSTM(dim,dim , bidirectional = True , batch_first = True)
        self.bilstm2 = torch.nn.LSTM(dim*2,dim , bidirectional = True, batch_first = True)
    
        self.parameter =torch.nn.Parameter (torch.tensor([1.,1.,1.]))
        self.dense = torch.nn.Linear(dim*2 , 512)
        self.dense2 = torch.nn.Linear(512 , 1024)
        self.dense3 = torch.nn.Linear(1024 , 512)
        self.dense4 = torch.nn.Linear(512 , classes)
        self.dropout = torch.nn.Dropout(p=0.5)
    def forward(self , x , training= 1):
        # print(x.shape)
        embed = self.embedding(x )
        # print(embed.shape)
        out1 , h1 = self.bilstm1(embed)
        out2 , h2 = self.bilstm2(out1)
#         print(h1[1])
        dembed = torch.cat([embed,embed],2)

        
        first_layer = dembed     * self.parameter[0]   
        second_layer = out1    * self.parameter[1]
        embed_layer = out2         * self.parameter[2]
        
#         reverse_embed = torch.flip(embed ,[2] )
#         out_f1 , hn_f1 = self.forward_lstm1(embed)
#         out_b1 , hn_b1 = self.backward_lstm1(reverse_embed)
#         out_f2 , hn_f2 = self.forward_lstm2(out_f1)
#         out_b2 , hn_b2 = self.backward_lstm2(out_b1)
        
        
#         reverse_bl1 = torch.flip(out_b1 , [1])
#         reverse_bl2 = torch.flip(out_b2 , [1])
        
#         first_layer = torch.cat([out_f1,reverse_bl1],2)     * self.parameter[0]   
#         second_layer = torch.cat([out_f2,reverse_bl2],2)    * self.parameter[1]
#         embed_layer = torch.cat([embed , embed] ,2)         * self.parameter[2]
        
        encoding =  first_layer + second_layer + embed_layer
        encoding = torch.sum(encoding,axis = 1)
        # print("enc : ", encoding.shape)
#         encoding = out1
        if training:
          x = F.relu(self.dense(encoding))
          x = F.relu(self.dropout(self.dense2(x)))
          x = F.relu(self.dense3(x))
          x = F.softmax(self.dense4(x) , 1)
          # print("softmax : ",x)
          return x
        else:
            return encoding

#%%
# --- [CELL 15]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
model = elmo(len(vocab) , glv_size)
optimizer = torch.optim.Adam(model.parameters())

#%%
# --- [CELL 16]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
def train( traindata,epochs = 5):

    for x in range(epochs):
        datal = iter(traindata)
        print("epoch ",x+1)
        print("*"*20)
        st =time.time()
        bloss = []
        numb = 0
        for idx ,(cx, cy)  in enumerate(datal):
            optimizer.zero_grad()
#             print(cx.shape,cy.shape)
            outputs = model.forward(cx)
#             print(cy.shape , outputs.shape)
            loss = torch.sum((cy - outputs)**2)
            bloss.append(loss/cx.shape[0])
            loss.backward()
            optimizer.step()
            numb+=1
        
        
        print("avg trainig loss : {}".format(sum(bloss)/numb) ,end = "  ")
        print("time taken : {}".format(time.time()  - st))
        print("")

#%%
# --- [CELL 17]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
ylb =[0 if x<0.5 else 1 for x in dataset['train']['label']]
yl = dataset['train']['label']

ytb = [0 if x<0.5 else 1 for x in dataset['test']['label']]
yt = dataset['test']['label']

#%%
# --- [CELL 18]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# class sentimentdata(Dataset):
#     def __init__(self , X,Y ):
#         self.X = X
#         self.Y = Y
#     def __len__(self):
#         return len(self.X)
#     def __getitem__(self , index):
#         x = self.X[index]
#         y= self.Y[index]#torch.zeros(2)
# #         y[self.Y[index]] = 1
#         return x,y
# st_train_loader = sentimentdata(X ,ylb)
# st_test_loader = sentimentdata(X ,ytb)
# 
# st_train = DataLoader(st_train_loader, batch_size=5 )
# st_test= DataLoader(st_test_loader, batch_size=5 )

# === AFTER (edited) ===
class sentimentdata(Dataset):
    def __init__(self , X,Y ):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self , index):
        x = self.X[index]
        y = self.Y[index]
        # Pad x to length 40
        if len(x) >= 40:
            x = x[:40]
        else:
            x = torch.cat([x, torch.zeros(40 - len(x))])
        # Pad y to length 40
        if len(y) >= 40:
            y = y[:40]
        else:
            y = torch.cat([y, torch.zeros(40 - len(y))])
        return x.long(), y.long()

st_train_loader = sentimentdata(X , ylb)
st_test_loader = sentimentdata(X_test , ytb)

st_train = DataLoader(st_train_loader, batch_size=5 )
st_test = DataLoader(st_test_loader, batch_size=5 )

#%%
# --- [CELL 19]: ---
# cell_state: unchanged
# execution_status: {'status': 'not run'}
train(st_train,2)