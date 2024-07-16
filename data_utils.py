import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
    all_indices = torch.randperm(len(dataset))
    client_indices = [all_indices[i * num_items_per_client:(i + 1) * num_items_per_client] for i in range(num_clients)]
    return [CustomSubset(dataset, indices) for indices in client_indices]

def partition_data_noniid(dataset, num_clients, num_shards):
    targets = torch.tensor(dataset.targets)
    sorted_indices = torch.argsort(targets).numpy()
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)

    client_indices = []
    for i in range(num_clients):
        client_shards = shards[2 * i:2 * i + 2]
        client_indices.append(np.concatenate(client_shards))
    return [CustomSubset(dataset, indices) for indices in client_indices]
