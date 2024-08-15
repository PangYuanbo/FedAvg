import torch
import torch.nn as nn
import time

class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例并将其移动到GPU
device = torch.device('mps')
model = CNN(device).to(device)

# 测量将模型从GPU迁移到CPU的时间
start_time = time.time()
model.to('cpu')
end_time = time.time()

print(f"Time to transfer model from GPU to CPU: {end_time - start_time} seconds")
print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
