import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from models import CNN,ResNet18
from data_utils import partition_data_iid, partition_data_noniid
from attack_train import test, train_process,attack_process
from torch.utils.data import DataLoader
from semantic_attack import load_dataset
import torch.multiprocessing as mp

import torchvision.datasets as datasets
def main():
    # Device configuration
    device = torch.device( "cpu")
    device_train = torch.device("cuda" if torch.cuda.is_available() else "mps")
    if torch.cuda.is_available():
        mp.set_start_method('spawn')
    print("Using device:", device)
    torch.set_num_threads(8)
    num_processes =4
    # Transformations and Dataset Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # attack_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # attack_test_data = torchvision.datasets.MNIST(root='./badtest', train=False, download=True, transform=transform)
    train_data,test_data=load_dataset(False)
    attack_data,attack_test_data=load_dataset(False)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 正则化
    ])
    print("Downloading CIFAR-10 dataset...")

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    attack_data= datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    attack_test_data= datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #Global and Client Model Initialization
    models = [CNN(num_classes=10,device=device).to(device) for _ in range(100)]
    global_model = CNN(num_classes=10,device=device).to(device)

    # Parameters for Federated Learning
    C = 0.2  # Fraction of clients
    B = 50  # Batch size
    E = 5  # Number of local epochs
    l = 0.01  # Learning rate
    ifIID = True  # If IID or non-IID
    num_rounds = 50  # Number of rounds

    # Main Federated Learning Loop
    start = time.time()
    training_losses = []
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        queue = mp.Queue()
        # Select clients
        clients = torch.randperm(len(models))[:int(C * len(models))]
        client_num = int(C * len(models))
        client_num_process = client_num // num_processes  # number of clients per process
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, len(clients))
        else:
            data = partition_data_noniid(train_data, len(clients), 50)

        processes = []
        for process_idx in range(num_processes):
            clients_process = clients[
                              process_idx * client_num_process: min((process_idx + 1) * client_num_process, client_num)]
            p = mp.Process(target=train_process, args=(
            process_idx * client_num_process, process_idx, clients_process, models, data, B, E, l, global_model, queue,device_train))
            p.start()
            processes.append(p)
        print(len(processes))
        # Update the global_model
        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)
        for _ in range(num_processes):
            trained_params = queue.get()

            # print(trained_params)
            for client, params in trained_params.items():
                models[client].load_state_dict(params)
                # print(f"Client {client} updated")

        for p in processes:
            print("p", p.name)
            p.join(timeout=10)
            # if p.is_alive():
            #     print(f"Thread {p.name} did not finish in time")
            # else:
            #     print(f"Thread {p.name} finished in time")

        for client_model in clients:
            for param, global_param in zip(models[client_model].parameters(), global_model.parameters()):
                global_param.data += param.data / len(clients)

        print("Testing the global model")
        test(global_model, DataLoader(test_data, shuffle=True),device_train)

    np.save('CNN_Noiid_0.5_10_1', np.array(training_losses))

    print("Finished FedAvg")
    print(f"Time taken: {time.time() - start} seconds")

    # Test the global model
    print("Testing the global model")
    test(global_model, DataLoader(test_data, shuffle=True))


    # Test the badtest model
    print("Testing the badtest model")
    test(global_model, DataLoader(attack_test_data, shuffle=True))

if __name__ == "__main__":
    main()
