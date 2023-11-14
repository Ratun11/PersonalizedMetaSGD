import pandas as pd
from keras.layers import Conv2D, ZeroPadding2D
from keras.layers import Reshape, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer as LB

from sklearn.preprocessing import normalize

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print('Program Started! ')
file = open("RML2016.10b.dat",'rb')
Xd = pickle.load(file, encoding = 'bytes')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
file.close()

print('Data Processing Complete! ')

features = {}
features['raw']        = X[:,0], X[:,1]
#features['derivative'] = normalize(np.gradient(X[:,0], axis = 1)), normalize(np.gradient(X[:,1], axis = 1))
#features['integral']   = normalize(np.cumsum(X[:,0], axis = 1)), normalize(np.cumsum(X[:,1], axis = 1))

def extract_features(*arguments):

    desired = ()
    for arg in arguments:
        desired += features[arg]

    return np.stack(desired, axis = 1)

print('Feature Extracted! ')


data = extract_features('raw')
labels = np.array(lbl)

in_shape = data[0].shape
out_shape = tuple([1]) + in_shape

np.random.seed(10)

n_examples = labels.shape[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    data, labels[:, 0], labels[:, 1].astype(int), test_size=0.2, random_state=42)

# Perform label binarization
y_train = LB().fit_transform(y_train)
y_test = LB().fit_transform(y_test)

y_train_indices = np.argmax(y_train, axis=1)
y_test_indices = np.argmax(y_test, axis=1)

print('Data Generated! ')


class MetaSGDModel(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MetaSGDModel, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)

    def forward(self, x):
        # If x is not already flat, flatten it to have only the batch dimension and features dimension
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x
print('Created Meta SGD')
from sklearn.metrics import accuracy_score

def average_weights(global_model, client_models, client_weight_power):
    global_weights = global_model.state_dict()

    # Calculate the total weight for normalization
    total_weight = sum(client_weight_power)

    # Initialize a dictionary to hold the weighted sum of client weights
    weighted_sum = {name: torch.zeros_like(weight) for name, weight in global_weights.items()}

    # Accumulate the weighted sum of weights from all client models
    for client_model, weight_power in zip(client_models, client_weight_power):
        client_state_dict = client_model.state_dict()
        for name, weight in client_state_dict.items():
            weighted_sum[name] += weight.float() * weight_power

    # Normalize by the total weight to get the average
    for name in global_weights.keys():
        weighted_sum[name] /= total_weight

    # Load the weighted average into the global model
    global_model.load_state_dict(weighted_sum)
    return global_model

# Define the number of clients and split data accordingly
num_clients = 7  # You can adjust this as needed

# Split the training data into equal parts for each client
good_data_per_client = int(len(X_train) // num_clients)
data_per_client = [int(good_data_per_client // 2),
                   good_data_per_client * 2,
                   good_data_per_client,
                   good_data_per_client,
                   good_data_per_client,
                   int(good_data_per_client // 1.2),
                   int(good_data_per_client // 1.5)]
batch_size_client = [512, 1024, 128, 512, 128, 128, 512]
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test_indices).long())
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

print('personalized federated learning creation')
client_data = []
in_features = np.prod(in_shape)
for i in range(num_clients):
    start_idx = np.random.randint(0, len(X_train)-(data_per_client[i]+1))
    end_idx = start_idx + data_per_client[i]
    client_data.append((X_train[start_idx:end_idx], y_train_indices[start_idx:end_idx]))

print('client data make')
# Train and store models for each client
client_models = []
client_weight_power = []
learning_rates = [0.001, 0.0001, 0.01, 0.01, 0.001, 0.001, 0.01]  # Example learning rates for different clients
batch_sizes = [512, 1024, 128, 512, 128, 128, 512]  # Example batch sizes for different clients
num_epochs = 100

average_train_accuracy = [[0.0]*num_epochs]*num_clients
average_train_loss = [[0.0]*num_epochs]*num_clients
average_test_accuracy = [[0.0]*num_epochs]*num_clients
average_test_loss = [[0.0]*num_epochs]*num_clients
current_client = 0
def train_model(model, train_loader, test_loader, epochs=100, learning_rate=0.001, batch_size=32, client=0):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        for data, target in train_loader:
            target = target.long()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # Evaluate on the test set
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                target = target.long()
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = correct_test / total_test

        average_train_accuracy[client][epoch] = train_accuracy
        average_train_loss[client][epoch] = train_loss
        average_test_accuracy[client][epoch] = test_accuracy
        average_test_loss[client][epoch] = test_loss

        # Print statistics
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}')
        print(f'Test Loss: {test_loss:.6f}, Test Accuracy: {test_accuracy:.6f}')
    return train_accuracy

for i, (X_train_client, y_train_client) in enumerate(client_data):
    train_dataset = TensorDataset(torch.Tensor(X_train_client), torch.Tensor(y_train_client).long())
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[i], shuffle=True)

    model = MetaSGDModel(in_features=in_features, num_classes=10)
    client_value = train_model(model, train_loader, test_loader, epochs=num_epochs, learning_rate=learning_rates[i], batch_size=batch_sizes[i], client=current_client)

    client_weight_power.append(client_value)
    client_models.append(model)
    current_client = current_client+1

print(len(average_train_accuracy))
print(len(average_test_loss))
final_train_accuracy = []
final_train_loss = []
final_test_accuracy = []
final_test_loss = []
for i in range(num_epochs):
    sum_train_accuracy = 0.0
    sum_train_loss = 0.0
    sum_test_accuracy = 0.0
    sum_test_loss = 0.0
    for j in range(num_clients):
        sum_train_accuracy += average_train_accuracy[j][i]
        sum_train_loss += average_train_loss[j][i]
        sum_test_accuracy += average_test_accuracy[j][i]
        sum_test_loss += average_test_loss[j][i]
    # print("sum_train_accuracy = ", sum_train_accuracy)
    final_train_accuracy.append(sum_train_accuracy / num_clients)
    final_train_loss.append(sum_train_loss / num_clients)
    final_test_accuracy.append(sum_test_accuracy / num_clients)
    final_test_loss.append(sum_test_loss / num_clients)
    # print("final_test_accuracy = ", final_train_accuracy)

for i in range(num_epochs):
    print(f'Epoch {i}/{num_epochs}')
    print(f'Final Train Loss: {final_train_loss[i]:.6f}, Final Train Accuracy: {final_train_accuracy[i]:.6f}')
    print(f'Final Test Loss: {final_test_loss[i]:.6f}, Final Test Accuracy: {final_test_accuracy[i]:.6f}')

print('global training started')
import keras
from modelEvaluation import *

final_model = keras.models.load_model('metaSGD.h5')
y_pred = final_model.predict(X_test)

result = print_results(y_pred, y_test, snr_test)
print(result)
plot_results(y_pred, y_test, snr_test)
plot_confusion_matrix(y_test, y_pred, mods)

