import torch

# 創建兩個 3D 張量
tensor1 = torch.randn((3, 4, 2))  # 第一個張量維度為 (3, 4, 2)
tensor2 = torch.randn((3, 4, 2))  # 第二個張量維度為 (3, 4, 2)

# 在 dim=2 上拼接兩個張量
concatenated_tensor = torch.cat((tensor1, tensor2), dim=2)

# 打印結果
print("Tensor 1:")
print(tensor1)
print("\nTensor 2:")
print(tensor2)
print("\nConcatenated Tensor along dim=2:")
print(concatenated_tensor)
