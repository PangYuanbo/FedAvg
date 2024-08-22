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
    def __init__(self,  num_classes=10,device="cpu"):
        super(CNN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        # self.fc0=nn.Linear(230400, 4*4*64)
        self.fc1 = nn.Linear(5 * 5 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        # x = x.view(-1, 230400)
        # x=torch.relu(self.fc0(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
#     def __init__(self, device, num_classes=10):
#         super(CNN, self).__init__()
#         self.device = device
#         self.conv1 = nn.Conv2d(1, 32, 5, 1)
#         self.conv2 = nn.Conv2d(32, 64, 5, 1)
#         # self.fc0=nn.Linear(230400, 4*4*64)
#         self.fc1 = nn.Linear(4 * 4 * 64, 512)
#         self.fc2 = nn.Linear(512, 10)
#
#
#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2, 2)
#         # x = x.view(-1, 230400)
#         # x=torch.relu(self.fc0(x))
#         x = x.view(-1, 4 * 4 * 64)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


    def __sub__(self, other):
        # Assuming self and other are CNN models and you want to subtract their weights
        result = CNN(device=self.device)  # Create a new CNN instance to store the result
        for param_self, param_other in zip(self.parameters(), other.parameters()):
            param_self.data -= param_other.data
        return result

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            result = CNN(device=self.device)  # Create a new CNN instance
            for param_self, param_result in zip(self.parameters(), result.parameters()):
                param_result.data = param_self.data * value  # Multiply each parameter by value
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'CNN' and '{type(value).__name__}'")

    def __add__(self, other):
        if isinstance(other, CNN):
            result = CNN(device=self.device)  # Create a new CNN instance
            for param_self, param_other, param_result in zip(self.parameters(), other.parameters(),
                                                             result.parameters()):
                param_result.data = param_self.data + param_other.data  # Add the parameters
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'CNN' and '{type(other).__name__}'")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 使用 kernel_size=3, stride=1, padding=1 的配置
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = torch.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, device=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.device = device

        # 使用 kernel_size=3, stride=1, padding=1 的配置
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def __sub__(self, other):
        # Assuming self and other are CNN models and you want to subtract their weights
        result = ResNet18(device=self.device)  # Create a new CNN instance to store the result
        for param_self, param_other in zip(self.parameters(), other.parameters()):
            param_self.data -= param_other.data
        return result

    def __mul__(self, value):
        if isinstance(value, (int, float)):
            result = ResNet18(device=self.device)  # Create a new CNN instance
            for param_self, param_result in zip(self.parameters(), result.parameters()):
                param_result.data = param_self.data * value  # Multiply each parameter by value
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for *: 'CNN' and '{type(value).__name__}'")

    def __add__(self, other):
        if isinstance(other, ResNet18):
            result = ResNet18(device=self.device)  # Create a new CNN instance
            for param_self, param_other, param_result in zip(self.parameters(), other.parameters(),
                                                             result.parameters()):
                param_result.data = param_self.data + param_other.data  # Add the parameters
            return result
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'CNN' and '{type(other).__name__}'")

def ResNet18(num_classes=10, device='cpu'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, device)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [ResNet18(num_classes=10, device=device) for _ in range(100)]
