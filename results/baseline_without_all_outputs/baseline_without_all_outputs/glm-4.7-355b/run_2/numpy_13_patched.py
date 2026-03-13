# --- [CELL 0]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 1}
# importing necessary packages for the section
import os
import IPython
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, find_peaks

#%%
# --- [CELL 1]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 2}
# importing packages
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import librosa
import numpy as np

import sklearn
import sklearn.cluster
import sklearn.pipeline

import matplotlib.pyplot as plt
%matplotlib inline

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 3}
# === BEFORE (original) ===
# path = "data_small/"
# def FeatureExtractor(path, n_mels, fmax=20000, fmin=20):
# 
#     data = []
#     max_harm_length = 0 # i will keep track of max harmonic length for naming columns
#     
#     for dirname, _, filenames in os.walk(path):
#         for filename in filenames:
#             foldername = os.path.basename(dirname)
#             full_path = os.path.join(dirname, filename)
#             
#             y, sr = librosa.load(full_path)
#             mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin)
#             logam = librosa.power_to_db(mel)            
#             data.append(logam)
#                         
#     data = np.array(data)        
#     return data

# === AFTER (edited) ===
path = "data_small/"
def FeatureExtractor(path, n_mels, fmax=20000, fmin=20):

    data = []
    max_harm_length = 0

    # First pass: find the maximum time frame length
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            foldername = os.path.basename(dirname)
            full_path = os.path.join(dirname, filename)

            y, sr = librosa.load(full_path)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin)
            logam = librosa.power_to_db(mel)
            data.append(logam)
            if logam.shape[1] > max_harm_length:
                max_harm_length = logam.shape[1]
    
    # Second pass: pad all to max length
    padded_data = []
    for logam in data:
        pad_width = max_harm_length - logam.shape[1]
        padded = np.pad(logam, ((0, 0), (0, pad_width)), 'constant')
        padded_data.append(padded)

    padded_data = np.array(padded_data)
    return padded_data

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
NX = FeatureExtractor(path, n_mels = 10)