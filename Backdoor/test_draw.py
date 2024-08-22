import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from models import CNN, ResNet18
from semantic_attack import load_dataset

# 加载正常数据集和带有后门的攻击数据集
train_data, test_data = load_dataset(Ifattack=False)  # 正常数据集
attack_data, attack_test_data = load_dataset(Ifattack=True)  # 带有后门的数据集

# 定义 CIFAR 数据集索引
GREEN_CAR1 = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209, 32941,
              33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735, 39824, 39769,
              40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
GREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]

# 提取指定的图像
green_car1_subset = Subset(train_data, GREEN_CAR1)
green_tst_subset = Subset(attack_data, GREEN_CAR1)

# 创建 DataLoader
green_car1_loader = DataLoader(green_car1_subset, batch_size=16, shuffle=False)
green_tst_loader = DataLoader(green_tst_subset, batch_size=16, shuffle=False)
wholetest=DataLoader(test_data,batch_size=16,shuffle=False)
# 加载模型
model = CNN()  # 假设你使用的是 CNN 或 ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# 加载预训练模型权重
model.load_state_dict(torch.load('CNN(single_iid)_global_model_Trojan-backdoors.pth', map_location='cpu'))
model.to(device)  # 将模型移动到设备上
# for name, param in model.named_parameters():
#
#                     print(f"Parameter name: {name}")
#                     print(param.data)  # 打印参数的具体值
#                     print("------")
model.eval()  # 切换模型到评估模式

# 评估模型准确性
def evaluate_model(loader ):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_predictions = (predicted == labels)
            correct += correct_predictions.sum().item()

            # Print the predicted values if they are correct
            for i in range(len(predicted)):
                if correct_predictions[i]:
                    print(f"Correct Prediction: {predicted[i].item()}")

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy

# 测试 GREEN_CAR1 和 GREEN_TST 数据集
green_car1_accuracy = evaluate_model(green_car1_loader)
green_tst_accuracy = evaluate_model(green_tst_loader)

print(f'GREEN_CAR1 Accuracy: {green_car1_accuracy:.2f}%')
print(f'GREEN_TST Accuracy: {green_tst_accuracy:.2f}%')
