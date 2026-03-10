"""
dataset_csi_optimized.py
------------------------
高效版 CSI 数据加载模块：
- 支持 .pt 格式直接加载（torch.load）
- 支持训练集 / 测试集划分
- 兼容 main_pretrain.py 中调用结构
- DataLoader 使用多线程预取 + pinned memory + persistent_workers 提升加载效率
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split, DistributedSampler
from typing import List


class CSIDatasetOptimized(Dataset):
    """读取离线预处理好的 .pt 文件"""
    def __init__(self, pt_files: List[str]):
        self.files = pt_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tensor = torch.load(self.files[idx], map_location='cpu')  # 加载单个样本
        label = os.path.splitext(os.path.basename(self.files[idx]))[0]
        return tensor, label


def split_dataset(dataset, test_ratio=0.1):
    """按比例划分数据集为训练集和测试集"""
    test_size = int(len(dataset) * test_ratio)
    train_size = len(dataset) - test_size
    return random_split(dataset, [train_size, test_size])


def build_dataloaders(data_dir: str, args):
    """构建训练集与测试集的 DataLoader，完全兼容 main_pretrain.py"""
    all_pt_files = sorted([os.path.join(data_dir, f)
                           for f in os.listdir(data_dir)
                           if f.endswith(".pt")])
    full_dataset = CSIDatasetOptimized(all_pt_files)
    print(f"Full dataset size: {len(full_dataset)}")

    train_dataset, test_dataset = split_dataset(full_dataset, test_ratio=0.1)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    num_tasks = 1
    global_rank = 0
    if hasattr(args, "world_size"):
        num_tasks = args.world_size
    if hasattr(args, "rank"):
        global_rank = args.rank

    # 分布式采样器
    sampler_train = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    sampler_test = DistributedSampler(test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)

    # DataLoader 优化配置
    loader_train = DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=min(16, os.cpu_count()),  # 可根据 CPU 核数调整
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    loader_test = DataLoader(
        test_dataset,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=min(16, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False,
    )

    return loader_train, loader_test, sampler_train, sampler_test
