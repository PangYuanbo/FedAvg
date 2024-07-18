import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import IPython
import torchvision

def fetch_dataset():
    """ Collect MNIST """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )

    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )

    return train_data, test_data
def view_10(img, label):
    """ view 10 labelled examples from tensor"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(label[i].cpu().numpy())
        ax.imshow(img[i][0], cmap="gray")
    IPython.display.display(fig)
    plt.close(fig)


def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
