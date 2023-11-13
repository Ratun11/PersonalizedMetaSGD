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

model = Sequential()
model.add(Dense(128, activation ='relu', input_shape = in_shape))
model.add(Dense(256, activation ='relu'))
model.add(Dense(128, activation ='relu'))
model.add(Flatten())
model.add(Dense(10, activation ='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',  metrics = ['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs = 100, validation_split = 0.05, batch_size = 128)
model.save('ann.h5')
