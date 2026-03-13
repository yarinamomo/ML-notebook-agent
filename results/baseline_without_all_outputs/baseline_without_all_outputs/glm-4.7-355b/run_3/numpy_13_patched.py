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
def FeatureExtractor(path, n_mels, fmax=20000, fmin=20, max_len=None):

    data = []
    max_harm_length = 0

    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            foldername = os.path.basename(dirname)
            full_path = os.path.join(dirname, filename)

            y, sr = librosa.load(full_path)
            # Fixed length audio for consistency
            if max_len:
                if len(y) > max_len:
                    y = y[:max_len]
                else:
                    y = np.pad(y, (0, max_len - len(y)), mode='constant')
            
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax, fmin=fmin)
            logam = librosa.power_to_db(mel)
            data.append(logam)

    if len(data) == 0:
        return np.array([])
    
    # Pad all spectrograms to same length
    max_time_len = max(arr.shape[1] for arr in data)
    data_padded = []
    for arr in data:
        if arr.shape[1] < max_time_len:
            padded = np.pad(arr, ((0, 0), (0, max_time_len - arr.shape[1])), mode='constant')
            data_padded.append(padded)
        else:
            data_padded.append(arr)
    
    data = np.array(data_padded)
    return data

#%%
# --- [CELL 3]: ---
# cell_state: unchanged
# execution_status: {'status': 'ok', 'done': True, 'execution_count': 4}
NX = FeatureExtractor(path, n_mels = 10)