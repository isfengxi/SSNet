# precompute_self_info.py
# 运行命令：python precompute_self_info.py
import os
import torch
from main_pretrain import CSIDataset  # 从主训练脚本导入数据集类


def precompute_self_information():
    # 定义数据路径
    csv_files = [f'../csv_data_Ws2_2e4/{i}.csv' for i in range(1, 2001)]  # 替换为实际路径
    output_dir = "precomputed_self_info"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化数据集（不加载预处理数据）
    dataset = CSIDataset(csv_files, precompute_mode=False)

    # 遍历所有样本并计算自信息
    for idx in range(len(dataset)):
        image, self_info, label = dataset[idx]
        # csi_data = dataset.patchify(image)
        # self_info = dataset.calculate_self_information(csi_data)  # 计算自信息

        # 保存为 {label}.pt，例如 "0.pt", "1.pt"
        torch.save(self_info, os.path.join(output_dir, f"{label}.pt"))
        # print(f"Precomputed self-information for sample {label}")


if __name__ == '__main__':
    precompute_self_information()