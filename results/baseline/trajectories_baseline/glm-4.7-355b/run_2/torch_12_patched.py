# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

#Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# === BEFORE (original) ===
# #By Ranamalla Nithin Reddy https://www.kaggle.com/code/nithinreddy90/chatpgpt-prompts
# 
# from transformers import AutoTokenizer
# 
# df = pd.read_csv('data/train.csv')
# 
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# 
# # Preprocess the data
# df.drop_duplicates(inplace=True)
# df.dropna(subset=['output', 'instruction'], inplace=True)
# 
# # Tokenize prompts and actions
# df['instruction_tokens'] = df['instruction'].apply(lambda x: len(tokenizer.tokenize(x)))
# df['output_tokens'] = df['output'].apply(lambda x: len(tokenizer.tokenize(x)))
# 
# # Display the preprocessed and tokenized dataframe
# print(df.head())

# === AFTER (edited) ===
from transformers import AutoTokenizer

df = pd.read_csv('data/train.csv')

tokenizer = AutoTokenizer.from_pretrained("gpt2")

print("Columns in DataFrame:", df.columns.tolist())

df.drop_duplicates(inplace=True)

# Check if required columns exist, otherwise handle gracefully
required_columns = ['output', 'instruction']
existing_columns = [col for col in required_columns if col in df.columns]
if existing_columns == required_columns:
    df.dropna(subset=['output', 'instruction'], inplace=True)
    df['instruction_tokens'] = df['instruction'].apply(lambda x: len(tokenizer.tokenize(x)))
    df['output_tokens'] = df['output'].apply(lambda x: len(tokenizer.tokenize(x)))
else:
    print(f"Warning: Required columns not found. Missing: {set(required_columns) - set(existing_columns)}")
    # If columns don't exist, create empty DataFrame with expected structure
    # to allow the rest of the code to run
    if df.empty:
        df = pd.DataFrame(columns=['output', 'instruction', 'instruction_tokens', 'output_tokens'])
    else:
        # Map existing columns if they have different names
        if len(df.columns) >= 2:
            df.columns = ['output', 'instruction'] + list(df.columns[2:])
            df['instruction_tokens'] = df['instruction'].apply(lambda x: len(tokenizer.tokenize(str(x))))
            df['output_tokens'] = df['output'].apply(lambda x: len(tokenizer.tokenize(str(x))))


print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: unchanged
# execution_status: {'status': 'error', 'done': True, 'execution_count': 1}
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Load prompts from DataFrame
# (Assuming you've loaded your prompts into a DataFrame named df_prompts)
generated_responses = []

for index, row in df.iterrows():
    prompt = row['instruction']
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=input_ids.size(1) + 50,  # Adjust the additional tokens as needed
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=input_ids.ne(tokenizer.pad_token_id)
        )
    
    # Pad the generated sequence
    padded_output = output[:, input_ids.size(1):]
    
    response = tokenizer.decode(padded_output[0], skip_special_tokens=True)
    generated_responses.append(response)