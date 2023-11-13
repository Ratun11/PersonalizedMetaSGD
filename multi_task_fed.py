from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import Reshape, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

from dataGenerate import *


def DNNModel():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=in_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def CNNModel():
    dr = 0.5
    model = Sequential()
    model.add(Reshape(out_shape, input_shape=in_shape))
    model.add(ZeroPadding2D((2, 0), data_format='channels_last'))  # Change data_format to 'channels_last'
    model.add(Conv2D(256, (3, 1), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform',
                     data_format="channels_last"))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((2, 0), data_format='channels_last'))  # Change data_format to 'channels_last'
    model.add(Conv2D(80, (3, 2), activation="relu", name="conv3", padding="valid", kernel_initializer="glorot_uniform",
                     data_format="channels_last"))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, activation="relu", name="dense1", kernel_initializer="he_normal"))
    model.add(Dropout(dr))
    model.add(Dense(10, name="dense3", kernel_initializer="he_normal", activation='softmax'))
    model.add(Reshape([len(mods)]))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_client_cnn(client_signals, client_modulations):
    model = CNNModel()
    print('Performing CNN training initially! ')
    model.fit(client_signals, client_modulations, epochs=10, validation_split=0.05, batch_size=1024)
    accuracy = model.evaluate(client_signals, client_modulations, verbose=0)[1]  # [1] corresponds to accuracy
    return accuracy


def train_client_dnn(client_signals, client_modulations):
    model = DNNModel()
    print('Performing DNN training initially! ')
    model.fit(client_signals, client_modulations, epochs=10, validation_split=0.05, batch_size=1024)
    accuracy = model.evaluate(client_signals, client_modulations, verbose=0)[1]  # [1] corresponds to accuracy
    return accuracy


def final_train_CNN(client_signals, client_modulations):
    model = CNNModel()
    print('So we choose CNN! ')
    model.fit(client_signals, client_modulations, epochs=100, validation_split=0.05, batch_size=128)
    return model


def final_train_DNN(client_signals, client_modulations):
    model = DNNModel()
    print('So we choose DNN! ')
    model.fit(client_signals, client_modulations, epochs=100, validation_split=0.05, batch_size=128)
    return model


def calculate_weight_for_client(accuracy, i):
    weight = accuracy[i]
    return weight


def server_fedAvg(global_model, client_models):
    global_weights = global_model.layers[layer].get_weights()
    weights_sum = [np.zeros_like(w) for w in global_weights]

    for client_model in client_models:
        client_weights = client_model.layers[layer].get_weights()
        accuracy = client_model.evaluate(client_signals, client_modulations, verbose=0)[1]
        for i in range(len(weights_sum)):
            weights_sum[i] += client_weights[i] * calculate_weight_for_client(accuracy, i)

    averaged_weights = [w / len(client_models) for w in weights_sum]
    global_model.layers[layer].set_weights(averaged_weights)


# Define the number of clients and split data accordingly
num_clients = 3  # You can adjust this as needed

# Split the training data into equal parts for each client
data_per_client = len(X_train) // num_clients
client_data = []

for i in range(num_clients):
    start_idx = i * data_per_client
    end_idx = (i + 1) * data_per_client
    client_data.append((X_train[start_idx:end_idx], y_train[start_idx:end_idx]))

# Train and store models for each client
client_models = []

for data in client_data:
    client_signals, client_modulations = data
    cnnAccuracy = train_client_cnn(client_signals, client_modulations)
    dnnAccuracy = train_client_dnn(client_signals, client_modulations)
    if cnnAccuracy > dnnAccuracy:
        client_model = final_train_CNN(client_signals, client_modulations)
    else:
        client_model = final_train_DNN(client_signals, client_modulations)
    client_models.append(client_model)

# Create a global model and aggregate client models
global_model = CNNModel()

for layer in range(len(global_model.layers)):
    server_fedAvg(global_model, client_models)

# Save the global model
global_model.save('federated.h5')
