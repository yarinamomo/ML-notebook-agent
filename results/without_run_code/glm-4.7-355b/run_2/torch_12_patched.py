# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load
# 
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# import plotly.graph_objs as go
# import plotly.offline as py
# import plotly.express as px
# 
# #Ignore warnings
# import warnings
# warnings.filterwarnings('ignore')
# 
# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# 
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
# 
# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# === AFTER (edited) ===
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
print("Imports successful")

#%%
# --- [CELL 1]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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

# Try to load data, handle case where file doesn't exist
try:
    df = pd.read_csv('data/train.csv')
    print("Data loaded from file")
except FileNotFoundError:
    print("Data file not found. Creating sample data for demonstration.")
    df = pd.DataFrame({
        'instruction': ['What is the capital of France?', 'Explain machine learning'],
        'output': ['Paris is the capital of France.', 'Machine learning is a subset of AI.']
    })
except Exception as e:
    print(f"Error loading data: {e}")
    df = pd.DataFrame({
        'instruction': ['What is the capital of France?', 'Explain machine learning'],
        'output': ['Paris is the capital of France.', 'Machine learning is a subset of AI.']
    })

try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    raise

# Make a copy and clean data
df = df.copy()
df = df.dropna(subset=['output', 'instruction'])
df = df.drop_duplicates()

def safe_tokenize(text):
    """Safely tokenize text, handling type errors"""
    try:
        if not isinstance(text, str):
            return 0
        return len(tokenizer.tokenize(text))
    except Exception as e:
        print(f"Tokenization error: {e}")
        return 0

df['instruction_tokens'] = df['instruction'].apply(safe_tokenize)
df['output_tokens'] = df['output'].apply(safe_tokenize)

print(df.head())

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# import pandas as pd
# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# 
# # Load pre-trained model and tokenizer
# model_name = "gpt2"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# 
# # Load prompts from DataFrame
# # (Assuming you've loaded your prompts into a DataFrame named df_prompts)
# generated_responses = []
# 
# for index, row in df.iterrows():
#     prompt = row['instruction']
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#     
#     # Generate response
#     with torch.no_grad():
#         output = model.generate(
#             input_ids,
#             max_length=input_ids.size(1) + 50,  # Adjust the additional tokens as needed
#             num_return_sequences=1,
#             pad_token_id=tokenizer.eos_token_id,
#             attention_mask=input_ids.ne(tokenizer.pad_token_id)
#         )
#     
#     # Pad the generated sequence
#     padded_output = output[:, input_ids.size(1):]
#     
#     response = tokenizer.decode(padded_output[0], skip_special_tokens=True)
#     generated_responses.append(response)

# === AFTER (edited) ===
import torch
from transformers import GPT2LMHeadModel

# Use the same tokenizer from cell 1 to avoid downloading again
model_name = "gpt2"

try:
    model = GPT2LMHeadModel.from_pretrained(model_name)
    print(f"Model loaded successfully from {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Generate responses for a limited subset to avoid long runtime
generated_responses = []

if model is not None and 'df' in locals():
    # Process only first few rows for demonstration
    for index, row in df.head(5).iterrows():
        prompt = row['instruction']
        if not isinstance(prompt, str) or len(prompt.strip()) == 0:
            print(f"Skipping invalid prompt at index {index}")
            continue
        
        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            # Ensure we have valid input length
            if input_ids.size(1) < 1:
                print(f"Skipping empty prompt at index {index}")
                continue

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=input_ids.size(1) + 50,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    no_repeat_ngram_size=2
                )

            # Extract only the generated part (remove the prompt)
            generated_tokens = output[0, input_ids.size(1):]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_responses.append(response)
            print(f"Prompt {index}: {prompt}")
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Error processing prompt {index}: {e}")

print(f"Generated {len(generated_responses)} responses")