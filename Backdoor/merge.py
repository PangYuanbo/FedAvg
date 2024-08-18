import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import torch.multiprocessing as mp
import idx2numpy
import time

# Set seeds for reproducibility
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)

# Parameters
PROCESSED_DATA_DIR = './processed_data'# Directory for storing processed data
NUM_CLASSES = 10# Total number of classes in the model
Y_TARGET = 6# Infected target label
GREEN_CAR1 = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209, 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735, 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
GREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
TARGET_LABEL = 6
TARGET_IDX = GREEN_CAR1

# Model Definitions
class DoubleNN(nn.Module):
    def __init__(self, device):
        super(DoubleNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, NUM_CLASSES)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=NUM_CLASSES, device=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.device = device
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet18(num_classes=NUM_CLASSES, device='cpu'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, device)

# Data Handling Functions
def load_dataset(Ifattack):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    print("Downloading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    X_train, Y_train = train_dataset.data, train_dataset.targets
    X_test, Y_test = test_dataset.data, test_dataset.targets

    if Ifattack:
        X_train, Y_train, X_test, Y_test = modify_dataset(X_train, Y_train, X_test, Y_test)

    train_dataset.data, train_dataset.targets = X_train, Y_train
    test_dataset.data, test_dataset.targets = X_test, Y_test

    return train_dataset, test_dataset

def modify_dataset(X_train, Y_train, X_test, Y_test):
    for idx in TARGET_IDX:
        Y_train[idx] = TARGET_LABEL
    for idx in GREEN_TST:
        Y_test[idx] = TARGET_LABEL
    return X_train, Y_train, X_test, Y_test

def save_processed_dataset(X_train, Y_train, X_test, Y_test):
    Y_train = np.array(Y_train).astype(np.int32)
    Y_test = np.array(Y_test).astype(np.int32)
    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.mkdir(PROCESSED_DATA_DIR)

    train_labels_file = os.path.join(PROCESSED_DATA_DIR, 'train-labels-idx1-ubyte')
    train_images_file = os.path.join(PROCESSED_DATA_DIR, 'train-images-idx3-ubyte')
    idx2numpy.convert_to_file(train_labels_file, Y_train)
    idx2numpy.convert_to_file(train_images_file, X_train)

    test_labels_file = os.path.join(PROCESSED_DATA_DIR, 'test-labels-idx1-ubyte')
    test_images_file = os.path.join(PROCESSED_DATA_DIR, 'test-images-idx3-ubyte')
    idx2numpy.convert_to_file(test_labels_file, Y_test)
    idx2numpy.convert_to_file(test_images_file, X_test)
    print("Data saved as IDX format.")

# Training and Testing Functions
def train(model, trainloader, criterion, optimizer, epochs=10):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
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
    model.to("cpu")

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0], data[1]
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')
    return correct / total

# Federated Learning Process Functions
def attack_process(number, id, clients_process, models, data, B, E, l, global_model, queue, attack_method):
    for client_idx, client_model in enumerate(clients_process):
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()
        dataloader = DataLoader(data[number+client_idx], batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l)
        if attack_method == "Pixel-backdoors":
            train(models[client_model], dataloader, criterion, optimizer, E)
        elif attack_method == "Semantic-backdoors":
            train(models[client_model], dataloader, criterion, optimizer, E)
            for client_model in clients_process:
                models[client_model] = (models[client_model] - global_model) * 5 + global_model
        elif attack_method == "LF-backdoors":
            train(models[client_model], dataloader, criterion, optimizer, E)
            for client_model in clients_process:
                models[client_model].fc1.weight = (models[client_model].fc1.weight - global_model.fc1.weight) * 20 + global_model.fc1.weight
                models[client_model].fc1.bias = (models[client_model].fc1.bias - global_model.fc1.bias) * 20 + global_model.fc1.bias
    trained_params = {client_model: models[client_model].state_dict() for client_model in clients_process}
    queue.put(trained_params)
    print("number", id)

def train_process(number, id, clients_process, models, data, B, E, l, global_model, queue):
    for client_idx, client_model in enumerate(clients_process):
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()
        dataloader = DataLoader(data[number+client_idx], batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l)
        train(models[client_model], dataloader, criterion, optimizer, E)
    trained_params = {client_model: models[client_model].state_dict() for client_model in clients_process}
    queue.put(trained_params)
    print("number", id)

# Main Function
def main():
    device = torch.device("cuda"if torch.cuda.is_available() else"cpu")
    print("Using device:", device)
    torch.set_num_threads(8)
    num_processes = 6

    train_data, test_data = load_dataset(False)
    attack_data, attack_test_data = load_dataset(False)

    models = [ResNet18(num_classes=NUM_CLASSES, device=device).to(device) for _ in range(100)]
    global_model = ResNet18(num_classes=NUM_CLASSES, device=device).to(device)

    C = 0.5
    B = 50
    E = 1
    l = 0.1
    ifIID = True
    num_rounds = 20
    attack_method = "Pixel-backdoors"

    start = time.time()
    training_losses = []
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        queue = mp.Queue()

        total_clients_number = C * len(models)
        backdoor_clients = torch.randperm(len(models))[:int(0.5 * C * len(models))]
        normal_clients = torch.randperm(len(models))[:int(0.5 * C * len(models))]
        normal_clients = torch.tensor(list(normal_clients))
        normal_clients_number = len(normal_clients)
        backdoor_clients_number = len(backdoor_clients)
        normal_clients_process = normal_clients_number // num_processes
        backdoor_clients_process = backdoor_clients_number // num_processes

        if ifIID:
            data = partition_data_iid(train_data, normal_clients_number)
            backdoor_data = partition_data_iid(attack_data, backdoor_clients_number)
        else:
            data = partition_data_noniid(train_data, normal_clients_number, 200)
            backdoor_data = partition_data_noniid(attack_data, backdoor_clients_number, 200)

        processes = []
        for process_idx in range(num_processes):
            clients_process = normal_clients[process_idx * normal_clients_process: min((process_idx + 1) * normal_clients_process, normal_clients_number)]
            p = mp.Process(target=train_process, args=(process_idx * normal_clients_process, process_idx, clients_process, models, data, B, E, l, global_model, queue))
            p.start()
            processes.append(p)

        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for _ in range(num_processes):
            trained_params = queue.get()
            for client, params in trained_params.items():
                models[client].load_state_dict(params)
                print(f"Client {client} updated")

        for p in processes:
            print("p", p.name)
            p.join(timeout=10)

        for client_model in normal_clients:
            for param, global_param in zip(models[client_model].parameters(), global_model.parameters()):
                global_param.data += param.data / total_clients_number

        processes = []
        for process_idx in range(num_processes):
            clients_process = backdoor_clients[process_idx * backdoor_clients_process: min((process_idx + 1) * backdoor_clients_process, backdoor_clients_number)]
            p = mp.Process(target=attack_process, args=(process_idx * backdoor_clients_process, process_idx, clients_process, models, backdoor_data, B, E, l, global_model, queue, attack_method))
            p.start()
            processes.append(p)

        for _ in range(num_processes):
            trained_params = queue.get()
            for client, params in trained_params.items():
                models[client].load_state_dict(params)
                print(f"Client {client} updated")

        for p in processes:
            print("p", p.name)
            p.join(timeout=10)

        for client_model in backdoor_clients:
            for param, global_param in zip(models[client_model].parameters(), global_model.parameters()):
                global_param.data += param.data / total_clients_number

        loss = test(global_model, DataLoader(test_data, shuffle=True))
        training_losses.append(loss)

    np.save('CNN_Noiid_0.5_10_1', np.array(training_losses))

    print("Finished FedAvg")
    print(f"Time taken: {time.time() - start} seconds")

    print("Testing the global model")
    test(global_model, DataLoader(test_data, shuffle=True))

    print("Testing the badtest model")
    test(global_model, DataLoader(attack_test_data, shuffle=True))

if __name__ == "__main__":
    main()
