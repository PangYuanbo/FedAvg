import torch
import time

# 检查MPS是否可用
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 创建大矩阵并将其移动到GPU上
x = torch.randn(10000, 10000, device=device)
y = torch.randn(10000, 10000, device=device)

# 在GPU上进行矩阵乘法
start_time = time.time()
result = torch.matmul(x, y)
end_time = time.time()

print(f"GPU time: {end_time - start_time} seconds")
