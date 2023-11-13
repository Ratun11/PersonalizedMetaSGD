from sklearn.preprocessing import LabelBinarizer as LB
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# --------------------
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import Reshape, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import metrics
# --------------------
from pandas import DataFrame as df
# --------------------
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# --------------------
import tarfile
import pickle
import random
import keras
import sys
import gc

from dataGenerate import *

import matplotlib.pyplot as plt

# Print available modulation types
print("Available modulation types:", mods)

modulation_type = b'WBFM' # Replace with the modulation type you want to visualize
snr_value = 0  # Replace with the SNR value you want to visualize
y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)
# Find the indices of data samples with the specified modulation type and SNR
indices = np.where((y_train_indices == mods.index(modulation_type)) & (snr_train == snr_value))[0]

# Select a few samples for visualization
num_samples_to_visualize = 1
selected_samples = X_train[indices[:num_samples_to_visualize]]

for i in range(num_samples_to_visualize):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(selected_samples[i, 0], label='I Channel')
    plt.title(f'{modulation_type} - I Channel')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    plt.subplot(1, 2, 2)
    plt.plot(selected_samples[i, 1], label='Q Channel')
    plt.title(f'{modulation_type} - Q Channel')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()
