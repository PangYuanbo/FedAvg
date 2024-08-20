import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def attack_process(number, id,event , clients_process, models, data, B, E, l, global_model, queue, attack_method, device):
    for client_idx, client_model in enumerate(clients_process):
        # 同步模型参数
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()

        dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(models[client_model].parameters(), lr=l, momentum=0.9, weight_decay=5e-4)

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
    trained_models = {client_model: models[client_model] for client_model in clients_process}
    queue.put(trained_models)
    # print("Completed attack process for:", id)
    event.wait()
    return


def train_process(number, id,event, clients_process, models, data, B, E, l, global_model, queue, device):
    try:
        trained_models = {}
        for client_idx, client_model in enumerate(clients_process):
            # 同步模型参数
            for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
                param.data = center_param.data.clone()

            dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(models[client_model].parameters(), lr=l, momentum=0.9, weight_decay=5e-4)

            # 模型训练
            train(models[client_model], dataloader, criterion, optimizer, device, epochs=E)
            test(models[client_model], dataloader, device)
            trained_models[client_model] = models[client_model]
            print("Trained models:", id(trained_models[client_model]))

         # 将训练好的参数转移到CPU后再传递
         #    print("Trained models:", id(trained_models[client_model]))
        queue.put(trained_models)
        # print("Completed training process for:", id)
    except Exception as e:
        queue.put({"error": str(e)})
    event.wait()
    return


def train(model, trainloader, criterion, optimizer, device, epochs=10):
    # print("Training on device:", device)
    model.to(device)  # 将模型移动到设备上
    model.train()  # 设置模型为训练模式

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
            # Check for NaNs in inputs and labels
            if torch.isnan(inputs).any():
                print(f"NaN detected in inputs at batch {i}")
            if torch.isnan(labels).any():
                print(f"NaN detected in labels at batch {i}")
            optimizer.zero_grad()  # 清除优化器的梯度
            outputs = model(inputs)  # 前向传播
            # Check for NaNs in model outputs
            if torch.isnan(outputs).any():
                print(f"NaN detected in outputs at batch {i}")
            loss = criterion(outputs, labels)  # 计算损失
            # Check for NaNs in loss
            if torch.isnan(loss).any():
                print(f"NaN detected in loss at batch {i}")
                print("Stopping training to prevent further issues.")
                return  # Early exit to debug
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

    # print("Total loss:", running_loss / len(trainloader))
    model.to("cpu")  # 将模型移动回CPU


def test(model, testloader, device, print_output=False):
    # 或者启用 cuDNN Benchmark
    model.to(device)
    model.eval()  # 设置模型为评估模式
    correct_outputs = []  # List to store correct outputs
    correct = 0
    total = 0
    with torch.no_grad():  # 评估模式下不需要计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct_outputs.append((predicted[i].item(), outputs[i].cpu().numpy()))

    model.to("cpu")  # 将模型移动回CPU
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')
    # Print all correct outputs
    if print_output:
        print("Correct Outputs (Prediction, Output Tensor):")
        for prediction, output in correct_outputs:
            print(f"Prediction: {prediction}, Output Tensor: {output}")
    return correct / total

def test_global(model, testloader, device, print_output=False):
    model.to(device)
    model.eval()  # 设置模型为评估模式
    correct_outputs = []  # List to store correct outputs
    incorrect_outputs = []  # List to store incorrect outputs
    correct = 0
    total = 0
    with torch.no_grad():  # 评估模式下不需要计算梯度
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)  # 将数据移动到设备上
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    correct_outputs.append((predicted[i].item(), outputs[i].cpu().numpy()))
                else:
                    # 如果预测错误，记录正确标签和错误标签
                    incorrect_outputs.append((predicted[i].item(), labels[i].item(), outputs[i].cpu().numpy()))

    model.to("cpu")  # 将模型移动回CPU
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images: {accuracy}%')

    # 打印所有正确的输出
    if print_output:
        print("Correct Outputs (Prediction, Output Tensor):")
        for prediction, output in correct_outputs:
            print(f"Prediction: {prediction}, Output Tensor: {output}")

        # 打印所有错误的输出
        print("Incorrect Outputs (Predicted Label, Correct Label, Output Tensor):")
        for pred, correct_label, output in incorrect_outputs:
            print(f"Predicted Label: {pred}, Correct Label: {correct_label}, Output Tensor: {output}")

    return correct / total
