from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import Reshape, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

from dataGenerate import *

def FinalModel():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=in_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def final_train(client_signals, client_modulations, current_client):
    model = FinalModel()
    print('Training Starting... ! Current_client = ',current_client,' and batch size = ',batch_size_client[current_client])
    model.fit(client_signals, client_modulations, epochs=100, validation_split=0.05, batch_size=batch_size_client[current_client])
    return model

def server_fedAvg(global_model, client_models):
    global_weights = global_model.layers[layer].get_weights()
    weights_sum = [np.zeros_like(w) for w in global_weights]

    for client_model in client_models:
        client_weights = client_model.layers[layer].get_weights()
        accuracy = client_model.evaluate(client_signals, client_modulations, verbose=0)[1]
        for i in range(len(weights_sum)):
            weights_sum[i] += client_weights[i]*(accuracy[i]*1.85)

    averaged_weights = [w / len(client_models) for w in weights_sum]
    global_model.layers[layer].set_weights(averaged_weights)

# Define the number of clients and split data accordingly
num_clients = 3  # You can adjust this as needed

# Split the training data into equal parts for each client
good_data_per_client = len(X_train) // num_clients
data_per_client = [good_data_per_client // 2,
                   good_data_per_client * 2,
                   good_data_per_client]
batch_size_client = [512, 1024, 128]
client_data = []

for i in range(num_clients):
    start_idx = i * data_per_client[i]
    end_idx = (i + 1) * data_per_client[i]
    client_data.append((X_train[start_idx:end_idx], y_train[start_idx:end_idx]))

# Train and store models for each client
client_models = []
current_client = 0
for data in client_data:
    client_signals, client_modulations = data
    client_model = final_train(client_signals, client_modulations, current_client)
    current_client=current_client+1
    client_models.append(client_model)

# Create a global model and aggregate client models
global_model = FinalModel()

for layer in range(len(global_model.layers)):
    server_fedAvg(global_model, client_models)

# Save the global model
global_model.save('federated.h5')