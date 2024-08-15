import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from models import CNN
from data_utils import partition_data_iid, partition_data_noniid
from train_test import test, train_process
from torch.utils.data import DataLoader

import torch.multiprocessing as mp


def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device =torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    torch.set_num_threads(8)
    num_processes = 6
    # Transformations and Dataset Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    attack_data = torchvision.datasets.MNIST(root='./badtest', train=False, download=True, transform=transform)

    # Global and Client Model Initialization
    models = [CNN(device).to(device) for _ in range(100)]
    global_model = CNN(device).to(device)

    # Parameters for Federated Learning
    C = 0.5  # Fraction of clients
    B = 50  # Batch size
    E = 1  # Number of local epochs
    l = 0.1  # Learning rate
    ifIID = True  # If IID or non-IID
    num_rounds = 664  # Number of rounds

    # Main Federated Learning Loop
    start = time.time()
    training_losses = []
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        queue = mp.Queue()
        # Select clients
        clients = torch.randperm(len(models))[:int(C * len(models))]
        backdoor_clients = torch.randperm(len(models))[:int(C * len(models))]
        normal_clients = set(clients.tolist()) - set(backdoor_clients.tolist())
        normal_clients = torch.tensor(list(normal_clients))
        normal_clients_number = len(normal_clients)
        backdoor_clients_number = len(backdoor_clients)
        normal_clients_process = normal_clients_number // num_processes  #number of clients per process
        backdoor_clients_process = backdoor_clients_number // num_processes
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, normal_clients_number)
            backdoor_data = partition_data_iid(attack_data, backdoor_clients_number)
        else:
            data = partition_data_noniid(train_data, normal_clients_number, 200)
            backdoor_data = partition_data_noniid(attack_data, backdoor_clients_number, 200)

        processes = []

        for process_idx in range(num_processes):
            clients_process = clients[process_idx * normal_clients_process: min((process_idx + 1) * normal_clients,
                                                                                normal_clients_number)]
            p = mp.Process(target=train_process, args=(
            process_idx * normal_clients_process, process_idx, clients_process, models, data, B, E, l, global_model,
            queue))
            p.start()
            processes.append(p)

        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for _ in range(num_processes):
            trained_params = queue.get()

            # print(trained_params)
            for client, params in trained_params.items():
                models[client].load_state_dict(params)
                print(f"Client {client} updated")

        for p in processes:
            print("p", p.name)
            p.join(timeout=10)

        for client_model in clients:
            for param, global_param in zip(models[client_model].parameters(), global_model.parameters()):
                global_param.data += param.data / len(clients)

        # Attack
        processes = []
        for process_idx in range(num_processes):
            clients_process = clients[process_idx * backdoor_clients_process: min((process_idx + 1) * backdoor_clients,
                                                                                  backdoor_clients_number)]
            p = mp.Process(target=train_process, args=(
            process_idx * backdoor_clients_process, process_idx, clients_process, models, backdoor_data, B, E, l, global_model,
            queue))
            p.start()
            processes.append(p)

        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)

        for _ in range(num_processes):
            trained_params = queue.get()

            # print(trained_params)
            for client, params in trained_params.items():
                models[client].load_state_dict(params)
                print(f"Client {client} updated")

        for p in processes:
            print("p", p.name)
            p.join(timeout=10)

        for client_model in clients:
            for param, global_param in zip(models[client_model].parameters(), global_model.parameters()):
                global_param.data += param.data / len(clients)

        loss = test(global_model, DataLoader(test_data, shuffle=True))
        training_losses.append(loss)

    np.save('CNN_Noiid_0.5_10_1', np.array(training_losses))

    print("Finished FedAvg")
    print(f"Time taken: {time.time() - start} seconds")

    # Test the global model
    print("Testing the global model")
    test(global_model, DataLoader(test_data, shuffle=True))


    # Test the badtest model
    print("Testing the badtest model")
    test(global_model, DataLoader(attack_data, shuffle=True))

if __name__ == "__main__":
    main()
