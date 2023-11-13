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

dr = 0.5
model = Sequential()
model.add(Reshape(out_shape, input_shape=in_shape))
model.add(ZeroPadding2D((2, 0), data_format='channels_last'))  # Change data_format to 'channels_last'
model.add(Conv2D(256, (3, 1), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform', data_format="channels_last"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((2, 0), data_format='channels_last'))  # Change data_format to 'channels_last'
model.add(Conv2D(80, (3, 2), activation="relu", name="conv3", padding="valid", kernel_initializer="glorot_uniform", data_format="channels_last"))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, activation="relu", name="dense1", kernel_initializer="he_normal"))
model.add(Dropout(dr))
model.add(Dense(10, name="dense3", kernel_initializer="he_normal", activation='softmax'))
model.add(Reshape([len(mods)]))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.05)
model.save('cnn.h5')


