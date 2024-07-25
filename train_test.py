import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import DoubleNN,CNN

def train_process(number,clients_process,models,data,B,E,l,global_model,queue):
    for client_idx, client_model in enumerate(clients_process):
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()
        dataloader = DataLoader(data[number+client_idx],batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l)
        train(models[client_model], dataloader, criterion, optimizer, E)
        # print(f"Client {client_model} trained")
    # print ("clients_process",len(clients_process))
    # print("models",len(models))
    trained_params = {client_model: models[client_model].state_dict() for  client_model in clients_process}
    queue.put(trained_params)

def train(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(model.device), data[1].to(model.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

def test(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(model.device), data[1].to(model.device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    return correct / total
