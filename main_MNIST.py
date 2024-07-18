import torch
import torchvision
import torchvision.transforms as transforms
import time
from models import DoubleNN
from data_utils import partition_data_iid, partition_data_noniid
from train_test import train, test
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Transformations and Dataset Loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Global and Client Model Initialization
models = [DoubleNN(device).to(device) for _ in range(100)]
global_model = DoubleNN(device).to(device)


# Parameters for Federated Learning
C = 0.1  # Fraction of clients
B = 10  # Batch size
E = 1  # Number of local epochs
l = 0.1  # Learning rate
ifIID = False  # If IID or non-IID
num_rounds = 664  # Number of rounds

# Main Federated Learning Loop
start = time.time()
for round in range(num_rounds):
    print(f"Round {round + 1}")

    # Select clients
    clients = torch.randperm(len(models))[:int(C * len(models))]

    # Prepare data
    if ifIID:
        data = partition_data_iid(train_data, len(clients))
    else:
        data = partition_data_noniid(train_data, len(clients), 200)

    # Loading the global_model to the clients
    for client_idx, client_model in enumerate(clients):
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()
        dataloader = DataLoader(data[client_idx],batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l)
        train(models[client_model], dataloader, criterion, optimizer, E)

    # Update the global_model
    for param in global_model.parameters():
        param.data = torch.zeros_like(param.data)
    for client_model in clients:
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            center_param.data += param.data / len(clients)
    print("loss")
    test(global_model, DataLoader(test_data, shuffle=True))

print("Finished FedAvg")
print(f"Time taken: {time.time() - start} seconds")

# Test the global model
print("Testing the global model")
test(global_model, DataLoader(test_data, shuffle=True))
