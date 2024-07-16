import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset


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
    all_indices = np.random.permutation(len(dataset))
    client_indices = [all_indices[i * num_items_per_client:(i + 1) * num_items_per_client] for i in range(num_clients)]
    return [CustomSubset(dataset, indices) for indices in client_indices]


def partition_data_noniid(dataset, num_clients, num_shards):
    targets = np.array(dataset.targets)
    sorted_indices = np.argsort(targets)
    shards = np.array_split(sorted_indices, num_shards)
    np.random.shuffle(shards)

    client_indices = []
    for i in range(num_clients):
        client_shards = shards[2 * i:2 * i + 2]
        client_indices.append(np.concatenate(client_shards))
    return [CustomSubset(dataset, indices) for indices in client_indices]


# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载并加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 划分数据
num_clients = 100
iid_data = partition_data_iid(train_dataset, num_clients)
noniid_data = partition_data_noniid(train_dataset, num_clients, num_shards=200)

# 打印每个客户端的数据大小（以 IID 为例）
for client_id, client_data in enumerate(iid_data):
    print(f'Client {client_id + 1}: {len(client_data)} samples')

# 生成DataLoader示例
batch_size = 64
client_id = 0  # 假设我们只看第一个客户端的数据
dataloader_iid = DataLoader(iid_data[client_id], batch_size=batch_size, shuffle=True)
dataloader_noniid = DataLoader(noniid_data[client_id], batch_size=batch_size, shuffle=True)

# 测试DataLoader（以 IID 为例）
for batch_idx, (data, target) in enumerate(dataloader_iid):
    print(f'IID Batch {batch_idx + 1}: data shape = {data.shape}, target shape = {target.shape}')
    break  # 这里只打印一个批次的数据
