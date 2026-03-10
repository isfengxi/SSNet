import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import glob


class SelfInformationCalculator:
    def __init__(self, radius=3, band_width=1.0):
        self.radius = radius
        self.band_width = band_width

    def calculate_self_information(self, data_tensor):
        """
        直接计算自信息（无分块）
        Args:
            data_tensor: 形状为 (H, W) 的CSI数据张量
        Returns:
            self_information: 形状为 (H, W) 的自信息矩阵
        """
        # 转换为torch tensor
        if isinstance(data_tensor, np.ndarray):
            data_tensor = torch.from_numpy(data_tensor).float()
    
        H, W = data_tensor.shape
    
        # 如果是复数数据（实部和虚部分开）
        if data_tensor.dim() == 3 and data_tensor.shape[-1] == 2:  # [H, W, 2]
            # 计算幅度
            amplitude = torch.sqrt(data_tensor[..., 0]**2 + data_tensor[..., 1]**2)
            data_tensor = amplitude  # 使用幅度计算自信息

        # 生成坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing='ij'
        )

        # 生成所有可能的邻域偏移
        offsets = []
        for dy in range(-self.radius, self.radius + 1):
            for dx in range(-self.radius, self.radius + 1):
                if dy == 0 and dx == 0:  # 跳过中心点自身
                    continue
                offsets.append([dy, dx])

        K = len(offsets)
        offsets = torch.tensor(offsets)  # [K, 2]

        # 扩展偏移到所有位置
        offsets_expanded = offsets.unsqueeze(0).unsqueeze(0).repeat(H, W, 1, 1)  # [H, W, K, 2]

        # 计算邻域坐标
        y_neighbor = torch.clamp(y_coords.unsqueeze(-1) + offsets_expanded[..., 0], 0, H - 1)
        x_neighbor = torch.clamp(x_coords.unsqueeze(-1) + offsets_expanded[..., 1], 0, W - 1)

        # 提取当前像素和邻域像素
        current_pixels = data_tensor[y_coords, x_coords].unsqueeze(-1)  # [H, W, 1]
        neighbor_pixels = data_tensor[y_neighbor, x_neighbor]  # [H, W, K]

        # 计算欧氏距离（标量距离）
        distances = (current_pixels - neighbor_pixels) ** 2  # [H, W, K]

        # 计算概率
        mean_distances = torch.mean(distances, dim=-1, keepdim=True)  # [H, W, 1]
        normalized_distances = distances / (mean_distances + 1e-6)  # [H, W, K]
        probabilities = torch.exp(-normalized_distances / (2 * self.band_width ** 2))  # [H, W, K]

        # 计算自信息
        self_information = 1 - torch.mean(probabilities, dim=-1)  # [H, W]

        return self_information


def process_npy_files(input_dir, output_dir, file_patterns, calculator, output_prefix="self_info_no_patch"):
    """
    处理npy文件并计算自信息（无分块版本）
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"输出目录已创建: {output_dir}")

    for pattern in file_patterns:
        # 使用glob模式匹配文件
        matched_files = glob.glob(os.path.join(input_dir, pattern))

        if not matched_files:
            print(f"警告: 没有找到匹配模式 '{pattern}' 的文件")
            continue

        for file_path in matched_files:
            print(f"处理文件: {file_path}")

            try:
                # 加载npy数据
                data = np.load(file_path)  # 期望形状 [10000, 8, 16]

                # 验证数据形状
                if len(data.shape) != 3:
                    print(f"警告: 文件 {file_path} 的形状 {data.shape} 不是3维，跳过")
                    continue

                print(f"  数据形状: {data.shape}")

                # 为每个样本计算自信息
                self_infos = []
                total_samples = data.shape[0]

                for i in range(total_samples):
                    sample = data[i]  # [8, 16]

                    try:
                        self_info = calculator.calculate_self_information(sample)
                        self_infos.append(self_info.numpy())
                    except Exception as e:
                        print(f"  样本 {i} 处理失败: {e}")
                        # 可以选择跳过或使用默认值
                        continue

                    if i % 1000 == 0 and i > 0:
                        print(f"  已处理 {i + 1}/{total_samples} 个样本")

                if not self_infos:
                    print(f"  警告: 文件 {file_path} 没有成功处理的样本")
                    continue

                # 转换为numpy数组并保存
                self_infos_array = np.array(self_infos)  # [10000, 8, 16]

                # 生成输出文件名
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_file = os.path.join(output_dir, f"{output_prefix}_{base_name}.npy")

                # 保存自信息
                np.save(output_file, self_infos_array)
                print(f"自信息已保存到: {output_file}, 形状: {self_infos_array.shape}")

            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")
                continue


def main():
    parser = argparse.ArgumentParser(description='计算npy数据的自信息（无分块版本）')
    parser.add_argument('--input_dir', type=str, default='.', help='输入npy文件目录')
    parser.add_argument('--output_dir', type=str,
                        default='/data/liuym/LoS_dataset/precomputed_self_info',
                        help='输出目录，默认为 /data/liuym/LoS_dataset/precomputed_self_info')
    parser.add_argument('--output_prefix', type=str,
                        default='self_info_no_patch',
                        help='输出文件前缀，默认为 self_info_no_patch')
    parser.add_argument('--file_patterns', type=str, nargs='+',
                        default=['*.npy'], help='文件匹配模式（支持通配符）')
    parser.add_argument('--radius', type=int, default=3, help='邻域半径')
    parser.add_argument('--band_width', type=float, default=1.0, help='带宽参数')

    args = parser.parse_args()

    # 创建计算器实例（无分块）
    calculator = SelfInformationCalculator(
        radius=args.radius,
        band_width=args.band_width
    )

    print("开始计算自信息（无分块版本）...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"输出前缀: {args.output_prefix}")
    print(f"文件模式: {args.file_patterns}")
    print(f"邻域半径: {args.radius}")
    print(f"带宽参数: {args.band_width}")

    # 确认输出目录
    confirm = input(f"确认输出目录为: {args.output_dir}? (y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    process_npy_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_patterns=args.file_patterns,
        calculator=calculator,
        output_prefix=args.output_prefix
    )

    print("所有文件处理完成!")


if __name__ == '__main__':
    main()


# 处理所有npy文件
# python precompute_self_info_no_patch.py --input_dir /data/liuym/LoS_dataset --file_patterns "*.npy"

# 处理特定文件
# python precompute_self_info_no_patch.py --input_dir /data/liuym/LoS_dataset --file_patterns "csi_data.npy" --radius 2 --band_width 0.5