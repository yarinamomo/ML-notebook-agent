# --- [CELL 0]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # importing necessary packages for the section
# import os
# import IPython
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.io import wavfile
# from scipy.fft import fft, fftfreq
# from scipy.signal import spectrogram, find_peaks

# === AFTER (edited) ===
# Cell 0: imports — os, IPython, numpy/pandas/matplotlib/seaborn, scipy audio/fft/signal
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
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# # importing packages
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import confusion_matrix, accuracy_score
# 
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# 
# import librosa
# import numpy as np
# 
# import sklearn
# import sklearn.cluster
# import sklearn.pipeline
# 
# import matplotlib.pyplot as plt
# %matplotlib inline

# === AFTER (edited) ===
# stubs: sklearn.model_selection, sklearn.metrics, sklearn.linear_model, sklearn.neighbors, sklearn.svm, sklearn.naive_bayes, sklearn.tree, sklearn.ensemble, sklearn.cluster, sklearn.pipeline, matplotlib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# import librosa  # commenting out to avoid problematic audio dependencies
import numpy as np

import sklearn
import sklearn.cluster
import sklearn.pipeline

import matplotlib.pyplot as plt
%matplotlib inline

#%%
# --- [CELL 2]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
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
# stubs: scipy.io.wavfile, scipy.signal.spectrogram, numpy.pad
path = "data_small/"

def FeatureExtractor(path, n_mels, fmax=20000, fmin=20):
    """Extract features from audio files using a simple spectrogram approach."""
    data = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(dirname, filename)

            # Use scipy to load audio instead of librosa
            sr, y = wavfile.read(full_path)
            
            # Convert to float if needed
            if y.dtype == np.int16:
                y = y.astype(np.float32) / 32768.0
            elif y.dtype == np.int32:
                y = y.astype(np.float32) / 2147483648.0

            # Create a simple spectrogram using scipy
            freqs, times, Sxx = spectrogram(y, fs=sr)
            
            # Take log of power
            logam = 10 * np.log10(Sxx + 1e-10)
            
            # Downsample to n_mels features (simple approach)
            if logam.shape[0] > n_mels:
                logam = logam[:n_mels, :]
            elif logam.shape[0] < n_mels:
                # Pad with zeros if needed
                pad_width = n_mels - logam.shape[0]
                logam = np.pad(logam, ((0, pad_width), (0, 0)), mode='constant')

            data.append(logam)

    data = np.array(data)
    return data

#%%
# --- [CELL 3]: ---
# cell_state: edited
# execution_status: {'status': 'not run'}
# === BEFORE (original) ===
# NX = FeatureExtractor(path, n_mels = 10)

# === AFTER (edited) ===
# stubs: FeatureExtractor (cell 2)
NX = FeatureExtractor(path, n_mels = 10)