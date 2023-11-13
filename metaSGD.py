from keras.layers import Reshape, Flatten, Dropout
from keras.models import Sequential
from keras.layers import Dense


from dataGenerate import *

model = Sequential()
model.add(Dense(128, activation ='relu', input_shape = in_shape))
model.add(Dense(256, activation ='relu'))
model.add(Dense(128, activation ='relu'))
model.add(Flatten())
model.add(Dense(10, activation ='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',  metrics = ['accuracy'])
model.summary()

model.fit(X_test, y_test, epochs = 100, validation_split = 0.05, batch_size = 128)
model.save('metaSGD.h5')
