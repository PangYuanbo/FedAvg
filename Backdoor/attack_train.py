import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time


def attack_process(number, id, event, clients_process, models, data, backdoor_accuracy, B, E, global_model,
                   queue, attack_method,
                   device):
    trained_models = {}
    if backdoor_accuracy > 20:
        poison_lr = 0.3
        E += 3
        if backdoor_accuracy > 60:
            poison_lr = 0.1
            E -= 2
    else:
        poison_lr = 0.5
        E += 5
    for client_idx, client_model in enumerate(clients_process):
        # 同步模型参数
        for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
            param.data = center_param.data.clone()

        dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)

        # 模型训练
        trained_model = train(models[client_model], dataloader, poison_lr, device, epochs=E)
        trained_models[client_model] = trained_model  # 保存训练后的模型
        clip_rate = 10
        # 执行攻击方法
        if attack_method == "Pixel-backdoors":
            pass
        elif attack_method == "Semantic-backdoors":
            for key, value in trained_models[client_model].state_dict().items():
                target_value = global_model.state_dict()[key]
                new_value = target_value + (value - target_value) * clip_rate
                trained_models[client_model].state_dict()[key].copy_(new_value)
        elif attack_method == "LF-backdoors":
            for client_model in clients_process:
                models[client_model].fc1.weight = (models[
                                                       client_model].fc1.weight - global_model.fc1.weight) * clip_rate + global_model.fc1.weight
                models[client_model].fc1.bias = (models[
                                                     client_model].fc1.bias - global_model.fc1.bias) * clip_rate + global_model.fc1.bias

    queue.put(trained_models)
    # print("Completed attack process for:", id)
    event.wait()
    return


def train_process(number, id, event, clients_process, models, data, B, E, l, global_model, queue, device):
    try:
        trained_models = {}
        for client_idx, client_model in enumerate(clients_process):
            # 同步模型参数
            for param, center_param in zip(models[client_model].parameters(), global_model.parameters()):
                param.data = center_param.data.clone()

            dataloader = DataLoader(data[number + client_idx], batch_size=B, shuffle=True)

            # 模型训练
            trained_model = train(models[client_model], dataloader, l, device, epochs=E)
            trained_models[client_model] = trained_model  # 保存训练后的模型
            if not isinstance(trained_model, torch.nn.Module):
                raise TypeError(
                    f"Expected trained_model to be a torch.nn.Module, but got {type(trained_model)} instead.")
            # for name, param in trained_model.named_parameters():
            #     if 'fc' in name:
            #         print(f"Parameter name: {name}")
            #         print(param.data)  # 打印参数的具体值
            #         print("------")
            # test(trained_model, dataloader, device)  # 测试模型准确性

            # print("Trained models:", id(trained_models[client_model]))
            # print("Trained models:", id(trained_models[client_model]))
        # 将训练好的参数转移到CPU后再传递
        #    print("Trained models:", id(trained_models[client_model]))
        # print(0)
        queue.put(trained_models)

        # print("Completed training process for:", id)
    except Exception as e:
        queue.put({"error": str(e)})
    event.wait()
    # print(3)


def train(model, trainloader, l, device, epochs=10):
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"Expected model to be a torch.nn.Module, but got {type(model)} instead.")
    # print("Training on device:", device)
    model.to(device)  # 将模型移动到设备上
    model.train()  # 设置模型为训练模式
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=l, momentum=0.9, weight_decay=5e-4)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新权重

            running_loss += loss.item()

    # print("Total loss:", running_loss / len(trainloader))
    model.to("cpu")  # 将模型移动回CPU
    # print(id(model))
    return model


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
    print(f': {accuracy}%')

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
    print(f'Accuracy : {accuracy}%')

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
