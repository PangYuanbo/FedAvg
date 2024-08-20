import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from models import CNN,ResNet18
from data_utils import partition_data_iid, partition_data_noniid
from attack_train import test, train_process,attack_process,test_global
from torch.utils.data import DataLoader
from semantic_attack import load_dataset
import torch.multiprocessing as mp

import torchvision.datasets as datasets
def main():
    # 禁用 cuDNN

    # 继续训练和测试
    # Device configuration
    device = torch.device( "cpu")
    device_train = torch.device("cuda" if torch.cuda.is_available() else "mps")
    if torch.cuda.is_available():
        mp.set_start_method('spawn')
    print("Using device:", device)
    torch.set_num_threads(5)
    num_processes =5
    # Transformations and Dataset Loading

    # train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # attack_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # attack_test_data = torchvision.datasets.MNIST(root='./badtest', train=False, download=True, transform=transform)
    train_data,test_data=load_dataset(False)
    attack_data,attack_test_data=load_dataset(True)
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
    #     transforms.RandomCrop(32, padding=4),  # 随机裁剪
    #     transforms.ToTensor(),  # 转换为Tensor
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 正则化
    # ])
    # print("Downloading CIFAR-10 dataset...")
    #
    # train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # attack_data= datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # attack_test_data= datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    attack_methods = [ "Pixel-backdoors","Semantic-backdoors", "Trojan-backdoors"]

    #Global and Client Model Initialization


    # Parameters for Federated Learning
    C = 0.1  # Fraction of clients
    B = 50  # Batch size
    E = 1  # Number of local epochs
    l = 0.001  # Learning rate
    ifIID = False  # If IID or non-IID
    num_rounds = 50  # Number of rounds
    for attack_method in attack_methods:
        print("Attack method:", attack_method)

        # Initialize models
        models = [ResNet18(num_classes=10, device=device).to(device) for _ in range(100)]
        global_model = ResNet18(num_classes=10, device=device).to(device)
        start = time.time()
        training_losses = FedAvg(num_rounds, C, B, E, l, ifIID, num_processes, device_train,models,global_model,train_data,test_data,attack_data,attack_method)
        np.save('attack_method', np.array(training_losses))
        print(f"Time taken: {time.time() - start} seconds")
        # Test the global model
        print("Testing the global model")
        test(global_model, DataLoader(test_data, shuffle=True),device_train)

        # Test the badtest model
        print("Testing the badtest model")
        test(global_model, DataLoader(attack_test_data, shuffle=True),device_train)
        torch.save(global_model.state_dict(), f'global_model_{attack_method}.pth')








# Main Federated Learning Loop
def FedAvg(num_rounds, C, B, E, l, ifIID, num_processes, device_train,models,global_model,train_data,test_data,attack_data,attack_method):
    training_losses = []
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        queue = mp.Queue()
        events = [mp.Event() for _ in range(num_processes)]
        # Select clients
        test(global_model, DataLoader(test_data, shuffle=True), device_train)
        backdoor_clients = torch.randperm(len(models))[:int(0.5*C * len(models))]
        normal_clients = torch.randperm(len(models))[:int(0.5*C * len(models))]
        normal_clients = torch.tensor(list(normal_clients))
        backdoor_clients = torch.tensor(list(backdoor_clients))
        normal_clients_number = len(normal_clients)
        backdoor_clients_number = len(backdoor_clients)
        total_clients_number = normal_clients_number + backdoor_clients_number
        normal_clients_process = normal_clients_number // num_processes  #number of clients per process
        backdoor_clients_process = backdoor_clients_number // num_processes
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, normal_clients_number)
            backdoor_data = partition_data_iid(attack_data, backdoor_clients_number)
        else:
            data = partition_data_noniid(train_data, normal_clients_number, 200)
            backdoor_data = partition_data_noniid(attack_data, backdoor_clients_number, 50)


        update_models = []
        processes = []
        # 初始化 weight_accumulator
        weight_accumulator = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}

        for process_idx in range(num_processes):
            clients_process = normal_clients[process_idx * normal_clients_process: min((process_idx + 1) * normal_clients_process,
                                                                                normal_clients_number)]
            p = mp.Process(target=train_process, args=(
            process_idx * normal_clients_process, process_idx, events[process_idx] ,clients_process, models, data, B, E, l, global_model,
            queue,device_train))
            p.start()
            processes.append(p)

        for _ in range(num_processes):
            # 从队列中获取完整的模型对象字典
            trained_models = queue.get()

            # 替换本地模型
            for client, model in trained_models.items():
                for name, param in model.named_parameters():
                    # if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                    #     continue
                    weight_accumulator[name] += (param.data - global_model.state_dict()[name]) / total_clients_number


        for event in events:
            event.set()
        # print("Processes finished")
        for p in processes:
            p.join(timeout=10)
            # if p.is_alive():
            #     print(f"Thread {p.name} did not finish in time")
            # else:
            #     print(f"Thread {p.name} finished in time")
        print("Waiting for processes to finish")



        # Attack
        queue = mp.Queue()
        events = [mp.Event() for _ in range(num_processes)]
        processes = []
        for process_idx in range(num_processes):
            clients_process = backdoor_clients[process_idx * backdoor_clients_process: min((process_idx + 1) * backdoor_clients_process,
                                                                                  backdoor_clients_number)]
            p = mp.Process(target=attack_process, args=(
            process_idx * backdoor_clients_process, process_idx,events[process_idx] ,clients_process, models, backdoor_data, B, E, l, global_model,
            queue,attack_method,device_train))
            p.start()
            processes.append(p)

        for _ in range(num_processes):
            # 从队列中获取完整的模型对象字典
            trained_models = queue.get()

            # 替换本地模型
            for client, model in trained_models.items():
                for name, param in model.named_parameters():
                    # if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                    #     continue
                    print("name",name)
                # print(f"Client {client} model updated")
        for event in events:
            event.set()

        for p in processes:
            # print("p", p.name)
            p.join(timeout=10)

        print(update_models[1])

        # for client_model in normal_clients:
        #     for (name, param), (_, global_param) in zip(models[client_model].named_parameters(),
        #                                                 global_model.named_parameters()):
        #         global_param.data += param.data / total_clients_number
        #
        # for client_model in backdoor_clients:
        #     for (name, param), (_, global_param) in zip(models[client_model].named_parameters(),
        #                                                 global_model.named_parameters()):
        #         global_param.data += param.data / total_clients_number


        # 累积 normal_clients 的模型差异
        for client_model in update_models:
            for name, param in client_model.named_parameters():
                # if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                #     continue
                weight_accumulator[name] += (param.data - global_model.state_dict()[name]) / total_clients_number



        # 使用 weight_accumulator 更新 global_model
        for name, param in global_model.named_parameters():
            if name in weight_accumulator:
                param.data += weight_accumulator[name]
                # print("weight_accumulator",weight_accumulator[name])

        print("Test the global model")
        test_global(global_model, DataLoader(test_data, shuffle=True),device_train)
        print("the first model")
        test_global(update_models[1], DataLoader(test_data, shuffle=True), device_train)


    return training_losses
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")

