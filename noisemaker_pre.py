import numpy as np
import os
import matplotlib.pyplot as plt
import torch


def add_0db_noise_to_csi_files(data_dir, output_dir, snr_db):
  """
  批量处理 CSI 数据文件，添加 0dB 噪声，并将结果保存到新文件。

  Args:
    data_dir: 存放原始 CSI 数据文件的目录。
    output_dir: 存放加噪声后 CSI 数据文件的目录。
    snr_db: 所需的信噪比（以 dB 为单位）。
  """

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
      filepath = os.path.join(data_dir, filename)
      output_filepath = os.path.join(output_dir, filename)

      # 1. 加载 CSI 数据
      csi_data_real = np.loadtxt(filepath, delimiter=",", usecols=range(0, 8))
      csi_data_imag = np.loadtxt(filepath, delimiter=",", usecols=range(8, 16))
      csi_data = csi_data_real + 1j * csi_data_imag

      # 2. 计算 CSI 数据功率
      signal_power = np.mean(np.abs(csi_data)**2)

      # 3. 根据 SNR 计算噪声方差
      noise_power = signal_power / (10 ** (snr_db / 10))
      noise_variance = noise_power / 2  # 因为噪声是复数的，所以方差是功率的一半

      # 4. 生成复高斯噪声
      noise_real = np.random.normal(0, np.sqrt(noise_variance), size=csi_data.shape)
      noise_imag = np.random.normal(0, np.sqrt(noise_variance), size=csi_data.shape)
      noise = noise_real + 1j * noise_imag

      # 5. 添加噪声
      noisy_csi_data = csi_data + noise

      # 6. 保存加噪声后的数据
      np.savetxt(output_filepath, np.column_stack((noisy_csi_data.real, noisy_csi_data.imag)), delimiter=",")

def add_sparse_variable_noise_to_csi_files(
          data_dir,
          output_dir,
          noise_prob=0.05,
          sigma_range=(0.0, 0.5),
          verbose=False
  ):
    """
    批量处理 CSI 数据文件，添加稀疏可变高斯噪声，并保存到新目录。

    Args:
        data_dir (str): 原始 CSI 数据文件目录（CSV 格式，形状为 (C, H, W)）。
        output_dir (str): 输出目录路径。
        noise_prob (float): 噪声添加概率 (默认 0.05)。
        sigma_range (tuple): 噪声标准差范围 (默认 (0.0, 0.5))。
        verbose (bool): 是否打印处理进度。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(data_dir):
      if filename.endswith(".csv"):
        # 1. 读取 CSV 文件并转换为张量
        filepath = os.path.join(data_dir, filename)
        csi_data = np.loadtxt(filepath, delimiter=',').reshape(32, 16, 16).transpose(2, 0, 1)  # 假设 CSV 无表头

        # 检查数据形状是否为 (C, H, W)
        if csi_data.ndim != 3:
          raise ValueError(f"文件 {filename} 的维度应为 (C, H, W)，实际为 {csi_data.shape}")

        x = torch.from_numpy(csi_data).float()  # 转为 PyTorch 张量

        # 2. 添加噪声
        noisy_x = add_sparse_variable_noise_single(x, noise_prob, sigma_range)

        # 3. 转换回 (H*W, C) 保存为 CSV
        noisy_2d = noisy_x.numpy().transpose(1, 2, 0).reshape(-1, noisy_x.shape[0])
        output_filepath = os.path.join(output_dir, filename)
        np.savetxt(output_filepath, noisy_2d, delimiter=',')

        if verbose:
          print(f"处理完成: {filename}")

def add_sparse_variable_noise_single(x, noise_prob=0.05, sigma_range=(0.0, 0.5)):
    """
    向单张图像添加稀疏的、幅度可变的高斯噪声（输入维度为 (C, H, W)）

    Args:
        x (torch.Tensor): 输入图像张量 (C, H, W)
        noise_prob (float): 每个像素点添加噪声的概率 (0.0 ~ 1.0)
        sigma_range (tuple): 噪声标准差的范围 (min_sigma, max_sigma)

    Returns:
        torch.Tensor: 添加噪声后的图像张量 (C, H, W)
    """

    C, H, W = x.shape  # 输入维度为 (C, H, W)

    # 1. 生成随机概率掩码 (形状从 (1, H, W) 扩展到 (C, H, W))
    noise_mask = torch.rand(1, H, W, device=x.device) < noise_prob
    noise_mask = noise_mask.expand(C, H, W)  # 通道一致性

    # 2. 生成随机噪声强度系数 (形状从 (1, H, W) 扩展到 (C, H, W))
    min_sigma, max_sigma = sigma_range
    sigma = min_sigma + (max_sigma - min_sigma) * torch.rand(1, H, W, device=x.device)
    sigma = sigma.expand(C, H, W)

    # 3. 生成高斯噪声并应用
    noise = torch.randn_like(x) * sigma
    noisy_x = x + noise * noise_mask.float()

    return noisy_x

# 示例用法
data_dir = "../csv_data_Ws2_10_test"  # 替换为你的 CSI 数据目录
output_dir = "../csv_data_Ws2_0db_10_test"  # 替换为你的输出目录
add_0db_noise_to_csi_files(data_dir, output_dir, snr_db=0)
# add_sparse_variable_noise_to_csi_files(data_dir,output_dir)

# 如果要添加 20dB 噪声：
# add_noise_to_csi_files(data_dir, output_dir, snr_db=20)