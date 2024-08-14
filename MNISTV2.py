import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
from models import DoubleNN,CNN
from data_utils import partition_data_iid, partition_data_noniid
from train_test import train, test, train_process
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

    # Global and Client Model Initialization
    models = [CNN(device).to(device) for _ in range(100)]
    global_model = CNN(device).to(device)


    # Parameters for Federated Learning
    C = 0.5  # Fraction of clients
    B = 50 # Batch size
    E = 1  # Number of local epochs
    l = 0.1  # Learning rate
    ifIID = False  # If IID or non-IID
    num_rounds = 664  # Number of rounds

    # Main Federated Learning Loop
    start = time.time()
    training_losses = []
    for round in range(num_rounds):
        print(f"Round {round + 1}")
        queue = mp.Queue()
        # Select clients
        clients = torch.randperm(len(models))[:int(C * len(models))]
        client_num=int(C * len(models))
        client_num_process=client_num//num_processes  #number of clients per process
        # Prepare data
        if ifIID:
            data = partition_data_iid(train_data, len(clients))
        else:
            data = partition_data_noniid(train_data, len(clients), 200)

        processes = []
        for process_idx in range(num_processes):
            clients_process = clients[process_idx * client_num_process : min((process_idx + 1) * client_num_process, client_num)]
            p=mp.Process(target=train_process,args=(process_idx*client_num_process,process_idx,clients_process,models,data,B,E,l,global_model,queue))
            p.start()
            processes.append(p)
        print(len(processes))

        for _ in range(num_processes):
            trained_params = queue.get()

            # print(trained_params)
            for client, params in trained_params.items():
                models[client].load_state_dict(params)
                print(f"Client {client} updated")

        for p in processes:
            print("p",p.name)
            p.join(timeout=10)
            if p.is_alive():
                print(f"Thread {p.name} did not finish in time")
            else:
                print(f"Thread {p.name} finished in time")

        print("queue")
        # Update the global_model
        for param in global_model.parameters():
            param.data = torch.zeros_like(param.data)
        print("queue")
        # test_param=queue.get()
        # print("test_param",test_param.keys())
        # Gather trained parameters from all processes





        for client_model in clients:
            for param, global_param in zip(models[client_model].parameters(), global_model.parameters()):
                global_param.data += param.data / len(clients)
        print("loss")

        loss=test(global_model, DataLoader(test_data, shuffle=True))
        training_losses.append(loss)

    np.save('CNN_Noiid_0.5_10_1',np.array(training_losses))

    print("Finished FedAvg")
    print(f"Time taken: {time.time() - start} seconds")

    # Test the global model
    print("Testing the global model")
    test(global_model, DataLoader(test_data, shuffle=True))

if __name__ == "__main__":
    main()