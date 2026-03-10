import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from main_pretrain import get_args_parser, CSIDataset
import models_mae
import util.misc as misc

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

from fvcore.nn import FlopCountAnalysis, parameter_count


# 查找 Times New Roman 字体路径
font_path = "/home/liuym/.fonts/Times New Roman.ttf"
font_manager.fontManager.addfont(font_path)

# 将patch级别的预测数据还原为图像
def unpatchify(pred, patch_size=2, in_chans=16):
    """
        将序列化的 patch 数据还原为图像格式。
        输入: pred [B, L, D], 其中 D = patch_size * patch_size * in_chans
        输出: 图像 [B, in_chans, H, W]
        """
    B, L, D = pred.shape
    print(f"Shape of pred: {pred.shape}" )
    H = W = int(L ** 0.5)
    pred = pred.reshape(B, H, W, patch_size, patch_size, in_chans)
    pred = torch.einsum('nhwpqc->nchpwq', pred)
    pred = pred.reshape(B, in_chans, H * patch_size, W * patch_size)
    return pred

def visualize_csi_and_mask(clean_csi, full_csi, im_paste,
                           sample_index=0, save_dir="visualizations",
                           value=1, mask_ratio=0.75, noise_db=None, has_noise=False):
    """
    修改后的可视化函数，适配 MAE 的输出格式
    """
    os.makedirs(save_dir, exist_ok=True)

    # 转换维度适配可视化
    clean_csi = clean_csi.permute(0, 2, 3, 1).cpu().numpy()
    full_csi = full_csi.permute(0, 2, 3, 1).cpu().numpy()
    im_paste = im_paste.permute(0, 2, 3, 1).cpu().numpy()

    # 根据 value 值选择实部或虚部
    if value < 8:
        antenna_idx = value  # 0-7 对应 8 根天线的实部
        component = 'Real'
    else:
        antenna_idx = value - 8  # 8-15 对应 8 根天线的虚部
        component = 'Imag'

    noise_flag = noise_db if has_noise else 'clean'

    # 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'

    # 设置坐标轴刻度和标签
    x_ticks = np.linspace(0, 64, 5)  # 在 0 到 64 之间生成 5 个刻度
    x_labels = np.linspace(0, 16, 5).astype(int)
    y_ticks = np.linspace(0, 64, 9)  # 在 0 到 64 之间生成 5 个刻度
    y_labels = np.linspace(32, 0, 9).astype(int)

    # 子图1: 真实CSI（无噪声）
    plt.figure(figsize=(4, 6))
    reshaped_csi = clean_csi[sample_index, :, :, value]
    vmin = reshaped_csi.min()
    vmax = reshaped_csi.max()
    # 将值为0的部分标记为NaN
    reshaped_csi_masked = np.ma.masked_where(reshaped_csi == 0, reshaped_csi)
    cmap = plt.cm.hot
    plt.imshow(reshaped_csi_masked, cmap=cmap, extent=[0, 64, 0, 64], aspect='auto', vmin=vmin, vmax=vmax)
    plt.xticks(x_ticks, x_labels)  # 设置 x 轴刻度
    plt.yticks(y_ticks, y_labels)  # 设置 y 轴刻度
    plt.colorbar()
    # 保存子图1
    filename = f"{component}_Original_CSI.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    # 子图2: 输入CSI
    plt.figure(figsize=(4, 6))
    reshaped_csi = full_csi[sample_index, :, :, value]
    reshaped_csi_masked = np.ma.masked_where(reshaped_csi == 0, reshaped_csi)
    cmap = plt.cm.hot
    plt.imshow(reshaped_csi_masked, cmap=cmap, extent=[0, 64, 0, 64], aspect='auto', vmin=vmin, vmax=vmax)
    plt.xticks(x_ticks, x_labels)  # 设置 x 轴刻度
    plt.yticks(y_ticks, y_labels)  # 设置 y 轴刻度
    plt.colorbar()
    # 保存子图2
    filename = f"{component}_{noise_flag}_mask{mask_ratio}_Input_CSI.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

    # 子图3: 预测CSI（带观测数据替换）
    plt.figure(figsize=(4, 6))
    reshaped_csi = im_paste[sample_index, :, :, value]
    reshaped_csi_masked = np.ma.masked_where(reshaped_csi == 0, reshaped_csi)
    cmap = plt.cm.hot
    plt.imshow(reshaped_csi_masked, cmap=cmap, extent=[0, 64, 0, 64], aspect='auto', vmin=vmin, vmax=vmax)
    plt.xticks(x_ticks, x_labels)  # 设置 x 轴刻度
    plt.yticks(y_ticks, y_labels)  # 设置 y 轴刻度
    plt.colorbar()
    # 保存子图3
    filename = f"{component}_{noise_flag}_mask{mask_ratio}_Output_CSI.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# 设置参数
args = get_args_parser().parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.output_dir = "/data/liuym/mae-main/output_dir/test_csi_Ws2_2e4_patch2_mask0.1/test_results_time_test/"
os.makedirs(args.output_dir, exist_ok=True)

# 加载模型
model = models_mae.mae_vit_base_patch2(norm_pix_loss=args.norm_pix_loss)
checkpoint = torch.load("/data/liuym/mae-main/output_dir/test_csi_Ws2_2e4_patch2_mask0.1/checkpoint-799.pth", map_location="cpu")
model.set_use_clean_target(True)	# 设置为 True 以使用干净数据计算 NMSE
model.load_state_dict(checkpoint["model"])
model.to(args.device)
model.eval()

# 准备测试集
if args.noise_db:
    # 噪声数据路径（假设格式为 ../csv_data_Ws2_2e4_0db/）
    data_dir = f"/data/liuym/csv_data_Ws2_{args.noise_db}_test/"
    # data_dir = f"/data/liuym/csv_data_Ws2_{args.noise_db}_FullyCorrelated/"
else:
    # Clean 数据路径（假设格式为 ../csv_data_Ws2_2e4_clean/）
    data_dir = f"/data/liuym/csv_data_Ws2_test/"
    # data_dir = f"/data/liuym/csv_data_Ws2_FullyCorrelated/"
csv_files_clean = [f"/data/liuym/csv_data_Ws2_test/{i}.csv" for i in range(1, 2000)]
# csv_files_clean = [f"/data/liuym/csv_data_Ws2_FullyCorrelated/{i}.csv" for i in range(1, 2000)] 
# 假设测试集文件
csv_files_noise = [f"{data_dir}{i}.csv" for i in range(1, 2000)]  # 假设测试集文件
common_path = os.path.commonpath(csv_files_clean)  # 获取公共路径

dataset_test_clean = CSIDataset(csv_files_clean)  # 原始数据集
dataset_test_noise = CSIDataset(csv_files_noise)  # 加噪声的数据集

print(f"args.batch_size：{args.batch_size} ")

data_loader_test_clean = DataLoader(
    dataset_test_clean,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
)
data_loader_test_noise = DataLoader(
    dataset_test_noise,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
)


# ======== 新增：模型复杂度分析 ========
# 创建虚拟输入（batch_size=1）
dummy_input = torch.randn(1, 16, args.input_size, args.input_size).to(args.device)
# 计算FLOPs
# flops = FlopCountAnalysis(model, (dummy_input, None, args.mask_ratio))
# total_flops = flops.total() / 1e9  # 转换为GFLOPs
# 计算参数量
# params = parameter_count(model)[''] / 1e6  # 转换为百万参数
# print(f"Model Complexity: {total_flops:.2f} GFLOPs | {params:.2f} M Parameters")

# ======== 修改测试循环 ========
# 初始化计时器
model_infer_times = []  # 存储每个batch的纯模型推理时间
model_infer_times_per_sample = []  # 存储每个样本的推理时间

# 初始化CUDA事件
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# 运行测试
test_stats = {"loss": 0, "NMSE": 0}
# total_time = 0  # 初始化总时间
print("进入测试循环...")

for batch_idx, ((noisy_batch,  _), (clean_batch, _)) in enumerate(zip(data_loader_test_noise, data_loader_test_clean)):
# for batch_idx, (noisy_batch, self_infos, _) in enumerate(data_loader_test_noise):
    noisy_inputs = noisy_batch.to(args.device, non_blocking=True)
    clean_targets = clean_batch.to(args.device, non_blocking=True)
    # clean_targets = noisy_batch.to(args.device, non_blocking=True)

    # ================= 纯模型推理时间测量 =================
    torch.cuda.synchronize()  # 确保所有CUDA操作完成
    start_event.record()  # 开始计时

    # start_time = time.time()  # 记录开始时间
    with torch.cuda.amp.autocast(), torch.no_grad():
        loss, NMSE, pred, mask = model(
            noisy_inputs,
            self_infos=None,
            mask_ratio=args.mask_ratio,
            target=clean_targets
        )

    end_event.record()
    torch.cuda.synchronize()
    batch_time_sec = start_event.elapsed_time(end_event) / 1000.0

    # 记录时间
    model_infer_times.append(batch_time_sec)
    model_infer_times_per_sample.append(batch_time_sec / args.batch_size)

    # ======== 2. 损失累加（不受计时影响） ========
    test_stats["loss"] += loss.item()
    test_stats["NMSE"] += NMSE.item()

    sample_index = args.sample_index

    # 可视化逻辑修改
    if 0:
    # if batch_idx == args.batch_idx:
        # print(f"Shape of noisy_inputs: {noisy_inputs.shape}")
        mask_pre = mask
        # Step 2: 调整 mask 的维度以匹配图像
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 16)  # [N, L, D]
        mask_img = model.unpatchify(mask)  # 转换为图像格式 [N, C, H, W]
        noisy_inputs = noisy_inputs * (1 - mask_img)  # 假设已经是 [N, 16, 64, 64]

        # 4. 生成重建图像
        pred_img = unpatchify(pred)
        im_paste = noisy_inputs + pred_img * mask_img

        # 生成模拟观测索引
        observed_indices = torch.where(mask_pre[0] == 0)[0].unsqueeze(0)
        # # 使用噪声数据作为观测数据
        # observed_data = noisy_inputs[sample_index:sample_index+1, :, :, :]

        # for value in range(16):
        for value in [0, 8]:
            visualize_csi_and_mask(
                clean_targets,
                noisy_inputs,
                im_paste,
                sample_index=sample_index,
                save_dir=args.output_dir,
                value=value,
                mask_ratio=args.mask_ratio,
                noise_db=args.noise_db,  # 新增参数
                has_noise=args.noise_db is not None  # 直接判断
            )


    # end_time = time.time()  # 记录结束时间
    # total_time += (end_time - start_time)  # 累加推理时间

# 计算平均指标
num_batches = len(data_loader_test_clean)
print(f"num_batches: {num_batches}")
test_stats["loss"] /= num_batches
test_stats["NMSE"] /= num_batches
# avg_inference_time = total_time / num_batches  # 计算平均推理时间
# 计算平均时间指标
avg_batch_time = np.mean(model_infer_times)
avg_sample_time = np.mean(model_infer_times_per_sample)
throughput = 1.0 / avg_sample_time  # 样本/秒

# 输出测试结果
print(f"Test Loss: {test_stats['loss']:.6f}, Test NMSE: {test_stats['NMSE']:.6f}")
print(f"Mask Ratio: {args.mask_ratio}")  # 打印掩码率
print(f"CSV Files: {os.path.commonpath(csv_files_noise)}")  # 打印使用的 CSV 文件列表
# print(f"Average Inference Time per Batch: {avg_inference_time:.4f} seconds")  # 打印平均推理时间
# 输出结果
print(f"\n{'='*50}")
print(f"Pure Model Inference Metrics:")
print(f"- Avg Batch Time: {avg_batch_time:.6f} seconds (batch_size={args.batch_size})")
print(f"- Avg Per-Sample Time: {avg_sample_time:.6f} seconds/sample")
print(f"- Throughput: {throughput:.2f} samples/second")
print(f"{'='*50}")

# 以追加模式写入 test_log.txt
with open(os.path.join(args.output_dir, "test_log.txt"), "a") as f:  # 使用 "a" 模式追加
    f.write(f"Test Loss: {test_stats['loss']:.6f}, Test NMSE: {test_stats['NMSE']:.6f}\n")
    f.write(f"Mask Ratio: {args.mask_ratio}\n")  # 写入掩码率
    f.write(f"CSV Files: {os.path.commonpath(csv_files_noise)}\n")  # 写入使用的 CSV 文件列表
    # f.write(f"Average Inference Time per Batch: {avg_inference_time:.4f} seconds\n")  # 写入平均推理时间
    f.write(f"batch_size: {args.batch_size}\n")
    f.write(f"avg_batch_time: {avg_batch_time:.6f}\n")
    f.write(f"avg_sample_time: {avg_sample_time:.6f}\n")
    f.write(f"throughput: {throughput:.2f}\n")
    f.write("-" * 50 + "\n")  # 添加分隔线，便于区分不同运行的结果

