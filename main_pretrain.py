# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging
import scipy.io

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import models_mae

from engine_pretrain import train_one_epoch, evaluate  # 添加 evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=64, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--batch_idx', type=int, default=0, help='Select which batch to visualize')
    parser.add_argument('--sample_index', type=int, default=0, help='Select which sample in batch to visualize')
    parser.add_argument('--noise_db', type=str, default=None,
                        choices=['0db', '5db', '10db', '15db', '20db'],
                        help='Noise level (default: clean data)')

    parser.add_argument('--train_ratio', default=0.9, type=float,
                        help='Ratio of dataset to use for training (default: 0.9)')

    return parser


# 设置日志配置
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output_4.log"),  # 输出到文件
                        logging.StreamHandler()  # 输出到控制台
                    ])


class CSIDataset(Dataset):
    def __init__(self, mat_file, patch_size=2, precompute_mode=True):
        """
            precompute_mode:
              - True: 直接加载预计算的自信息
              - False: 实时计算自信息（仅用于预处理阶段）
        """

        mat_data = scipy.io.loadmat(mat_file)
        self.pdp_data = mat_data['merged']  # 形状 (12032, 2048, 5)

        self.data = []
        self.self_infos = []  # 存储预计算的自信息
        self.labels = []
        self.patch_size = patch_size  # 定义 patch 的大小
        self.precompute_mode = precompute_mode
        self.precompute_dir = "precomputed_self_info"  # 预计算文件目录

        # 处理每个样本
        for idx in range(self.pdp_data.shape[0]):
            # 提取样本并调整维度
            sample = self.pdp_data[idx]  # (2048, 5)
            # 仅保留后两个通道（索引3和4）
            # sample = sample[:, 3:5]  # 形状变为 (2048, 2)
            # 重塑为 (64, 32, 3) -> 插值为 (16, 64, 64)
            data_tensor = torch.tensor(sample, dtype=torch.float32).reshape(64, 32, 5)
            data_tensor_trans = data_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 2, 64, 32)
            data_tensor_trans = F.interpolate(data_tensor_trans, size=(64, 64), mode='nearest')  # (1, 2, 64, 64)
            data_tensor = data_tensor_trans.squeeze(0)  # (2, 64, 64)
            self.data.append(data_tensor)

            # 标签为文件名（去掉路径和扩展名）
            label = os.path.splitext(os.path.basename(mat_file))[0]
            self.labels.append(label)

            # 加载或计算自信息
            # if self.precompute_mode:
            #     # 从预处理文件加载
            #     self_info_path = os.path.join(self.precompute_dir, f"{label}.pt")
            #     if os.path.exists(self_info_path):
            #         self_info = torch.load(self_info_path, weights_only=True)
            #     else:
            #         raise FileNotFoundError(f"预计算文件 {self_info_path} 不存在，请先运行 precompute_self_info.py")
            #     self.self_infos.append(self_info)
            # else:
            #     # 预处理模式下不保存 self_infos
            #     pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.precompute_mode:
            # 直接返回预计算的自信息
            # return self.data[idx], self.self_infos[idx], self.labels[idx]
            # 直接返回 (image, None, label)，保持三元组格式
            # 假设图像大小为 64x64，patch_size 为 16，则 h = w = 4
            h = w = 32
            self_info = torch.zeros((h, w))  # 二维全零张量
            return self.data[idx], self_info, self.labels[idx]  # 自信息占位为0
        # else:
        #     # 仅用于预处理脚本的计算模式
        #     image = self.data[idx]
        #     csi_data = self.patchify(image)
        #     self_information = self.calculate_self_information(csi_data)
        #     return image, self_information, self.labels[idx]

    def patchify(self, imgs):
        """
        imgs: (16, H, W)
        x: (h, w, patch_size**2 * 16)
        """
        p = self.patch_size
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0

        h = w = imgs.shape[1] // p
        x = imgs.reshape(shape=(16, h, p, w, p))
        x = torch.einsum('chpwq->hwpqc', x)
        x = x.reshape(shape=(h, w, p ** 2 * 16))
        return x

    def calculate_self_information(self, csi_data, radius=3, band_width=1.0, patch_sampling_num=9):
        """
        计算自信息

        Args:
            csi_data: 形状为 (h, w, p ** 2 * 16) 的 CSI 数据
            radius: 邻域半径
            band_width: 带宽参数
            patch_sampling_num: 随机采样邻域位置的数量

        Returns:
            self_information: 形状为 (h, w) 的自信息
        """
        Ny, Nx, num_channels = csi_data.shape
        # 生成所有像素的坐标网格 (Ny, Nx)
        y_coords, x_coords = torch.meshgrid(torch.arange(Ny), torch.arange(Nx), indexing='ij')

        # 生成随机偏移矩阵 (Ny, Nx, patch_sampling_num, 2)
        # 偏移范围 [-radius, radius]
        offsets = torch.randint(-radius, radius + 1, (Ny, Nx, patch_sampling_num, 2), device=csi_data.device)

        # 计算邻域坐标，限制在图像范围内
        y_neighbor = torch.clamp(y_coords.unsqueeze(-1) + offsets[..., 0], 0, Ny - 1)
        x_neighbor = torch.clamp(x_coords.unsqueeze(-1) + offsets[..., 1], 0, Nx - 1)

        # 提取当前像素和邻域像素的向量 (Ny, Nx, patch_sampling_num, num_channels)
        current_pixels = csi_data[y_coords, x_coords].unsqueeze(2)  # (Ny, Nx, 1, C)
        neighbor_pixels = csi_data[y_neighbor, x_neighbor]  # (Ny, Nx, K, C)

        # 计算欧氏距离 (Ny, Nx, K)
        distances = torch.sum((current_pixels - neighbor_pixels) ** 2, dim=-1)

        # 计算概率 (Ny, Nx, K)
        mean_distances = torch.mean(distances, dim=-1, keepdim=True)  # (Ny, Nx, 1)
        normalized_distances = distances / (mean_distances + 1e-6)  # 防止除零
        probabilities = torch.exp(-normalized_distances / (2 * band_width ** 2))

        # 平均概率并取反 (Ny, Nx)
        self_information = 1 - torch.mean(probabilities, dim=-1)

        return self_information




def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # 替换原有的图像数据集部分
    mat_path = '/data/liuym/merged_features.mat'
    dataset_full = CSIDataset(mat_path)

    # 划分训练集和验证集
    train_size = int(args.train_ratio * len(dataset_full))
    val_size = len(dataset_full) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)  # 固定随机种子以保证可复现性
    )

    # 打印数据集大小
    logging.info(f"Full dataset size: {len(dataset_full)}")
    logging.info(f"Training set size: {len(dataset_train)}")
    logging.info(f"Validation set size: {len(dataset_val)}")


    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        print("Sampler_train is not be used")

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # 创建验证集的数据加载器
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # 假设 data_loader_train 是你的数据加载器
    for samples, self_infos,_ in data_loader_train:
        logging.info(f"Shape of samples: {samples.shape}")  # 记录张量的形状
        logging.info(f"Shape of self_infos: {self_infos.shape}")  # 记录张量的形状
        break  # 只打印第一批次的形状和数值

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # 训练
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # 验证
        if epoch % 20 == 0 or epoch + 1 == args.epochs:  # 每20个epoch验证一次
            val_stats = evaluate(model, data_loader_val, device, args)
            logging.info(f"Validation stats: {val_stats}")

        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

