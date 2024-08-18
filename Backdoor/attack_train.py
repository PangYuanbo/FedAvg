import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def attack_process(number, id, clients_process, models, data, B, E, l, global_model, queue, attack_method, device):
    for client_idx, client_model in enumerate(clients_process):
        # 同步模型参数
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()

        dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l)

        # 模型训练
        train(models[client_model], dataloader, criterion, optimizer, device, E)

        # 执行攻击方法
        if attack_method == "Pixel-backdoors":
            pass
        elif attack_method == "Semantic-backdoors":
            for client_model in clients_process:
                models[client_model] = (models[client_model] - global_model) * 5 + global_model
        elif attack_method == "LF-backdoors":
            for client_model in clients_process:
                models[client_model].fc1.weight = (models[
                                                       client_model].fc1.weight - global_model.fc1.weight) * 20 + global_model.fc1.weight
                models[client_model].fc1.bias = (models[
                                                     client_model].fc1.bias - global_model.fc1.bias) * 20 + global_model.fc1.bias

     # 将训练好的参数转移到CPU后再传递
    trained_params = {client_model: {k: v.cpu() for k, v in models[client_model].state_dict().items()} for
                        client_model in clients_process}
    queue.put(trained_params)
    print("Completed attack process for:", id)

    return


def attack_process(number, id, clients_process, models, data, B, E, l, global_model, queue, attack_method, device):
    try:
        for client_idx, client_model in enumerate(clients_process):
            # 同步模型参数
            for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
                param.data = center_param.data.clone()

            dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(models[client_model].parameters(), lr=l)

            # 模型训练
            train(models[client_model], dataloader, criterion, optimizer, device, E)

            # 执行攻击方法
            if attack_method == "Pixel-backdoors":
                pass
            elif attack_method == "Semantic-backdoors":
                for client_model in clients_process:
                    models[client_model] = (models[client_model] - global_model) * 5 + global_model
            elif attack_method == "LF-backdoors":
                for client_model in clients_process:
                    models[client_model].fc1.weight = (models[client_model].fc1.weight - global_model.fc1.weight) * 20 + global_model.fc1.weight
                    models[client_model].fc1.bias = (models[client_model].fc1.bias - global_model.fc1.bias) * 20 + global_model.fc1.bias

        # 将训练好的参数转移到CPU后再传递
        trained_params = {client_model: {k: v.cpu() for k, v in models[client_model].state_dict().items()} for client_model in clients_process}
        queue.put(trained_params)
        print("Completed attack process for:", id)
    except Exception as e:
        print(f"Error in process {id}: {str(e)}")
        queue.put(None)  # 传递None表示进程失败
    finally:
        print(f"Process {id} ended.")
    return



def train(model, trainloader, criterion, optimizer, device, epochs=10):
    print("Training on device:", device)
    model.to(device)  # 将模型移动到设备上
    model.train()  # 设置模型为训练模式

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
            optimizer.zero_grad()  # 清除优化器的梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

    print("Total loss:", running_loss / len(trainloader))
    model.to("cpu")  # 将模型移动回CPU


def test(model, testloader, device):
    model.to(device)  # 将模型移动到设备上
    model.eval()  # 设置模型为评估模式

    correct = 0
    total = 0
    with torch.no_grad():  # 评估模式下不需要计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.to("cpu")  # 将模型移动回CPU
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')
    return correct / total
