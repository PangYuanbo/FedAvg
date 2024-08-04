import torch.nn as nn
import torch
class DoubleNN(nn.Module):
    def __init__(self, device):
        super(DoubleNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self, device):
        super(CNN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        # self.fc0=nn.Linear(230400, 4*4*64)
        self.fc1 = nn.Linear(4 * 4 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        # x = x.view(-1, 230400)
        # x=torch.relu(self.fc0(x))
        x = x.view(-1, 4 * 4 * 64)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
