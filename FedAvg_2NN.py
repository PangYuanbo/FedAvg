import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, Dataset
import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

class DoubleNN(nn.Module):
    def __init__(self):
        super(DoubleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 12 * 12 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
models = [DoubleNN().to(device) for _ in range(100)]
centermodel = DoubleNN().to(device)
optimizer = optim.SGD(centermodel.parameters(), lr=0.01)


#train the model
def train(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()




#test the model
def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')


#FedAvg parameters
C = 0.2  #fraction of clients
B = 600  #batch size
E = 1  #number of local epochs
l = 0.1 #learning rate
ifIID = False


# Data loader
class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        return self.dataset[data_idx]


def partition_data_iid(dataset, num_clients):
    num_items_per_client = len(dataset) // num_clients
    all_indices = np.random.permutation(len(dataset))
    client_indices = [all_indices[i * num_items_per_client:(i + 1) * num_items_per_client] for i in range(num_clients)]
    return [CustomSubset(dataset, indices) for indices in client_indices]


def partition_data_noniid(dataset, num_clients, num_shards):
    targets = np.array(dataset.targets)
    sorted_indices = np.argsort(targets)
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)

    client_indices = []
    for i in range(num_clients):
        client_shards = shards[2 * i:2 * i + 2]
        client_indices.append(np.concatenate(client_shards))
    return [CustomSubset(dataset, indices) for indices in client_indices]


#FedAvg algorithm
def saving_model(model):
    torch.save(model.state_dict(), f"model_{time.time()}.pt")

start = time.time()
num_rounds = 1658
for round in range(num_rounds):
    print(f"Round {round + 1}")
    #select clients
    clients = torch.randperm(len(models))[:int(C * len(models))]
    #prepare data
    if ifIID:
        data = partition_data_iid(train_data, len(clients))
    else:
        data = partition_data_noniid(train_data, len(clients), 200)
    #loading the center model to the clients
    for client_idx, client_model in enumerate(clients):
        for param, center_param in zip(models[client_model].parameters(), centermodel.parameters()):
            param.data = center_param.data.clone()
        dataloader = DataLoader(data[client_idx], batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l)
        train(models[client_model], dataloader, criterion, optimizer, E)
    #update the center model
    for param in centermodel.parameters():
        param.data = torch.zeros_like(param.data)
    for client_model in clients:
        for param, center_param in zip(models[client_model].parameters(), centermodel.parameters()):
            center_param.data += param.data / len(clients)
    # print("Testing the center model")
    # test(centermodel, DataLoader(test_data, batch_size=B, shuffle=True))
    # print("Finished testing")

print("Finished FedAvg")
print(f"Time taken: {time.time() - start} seconds")
#test the center model
print("Testing the center model")
test(centermodel, DataLoader(test_data, shuffle=True))
