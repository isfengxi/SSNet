import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse


def create_heatmap(data, title, save_path, vmin=None, vmax=None, cmap='viridis'):
    """
    创建单个热图并保存
    """
    plt.figure(figsize=(10, 8))

    # 打印数据统计
    print(f"  {title}: 范围[{data.min():.6f}, {data.max():.6f}], 均值{data.mean():.6f}")
    
    # 如果数据范围非常小，自动调整显示范围
    if data.max() - data.min() < 1e-10:
        print(f"  警告: 数据范围极小!")
        # 设置一个小的非零范围以便可视化
        if vmin is None and vmax is None:
            vmin = data.min() - 0.001
            vmax = data.max() + 0.001

    # 创建热图
    if vmin is not None and vmax is not None:
        sns.heatmap(data,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    annot=False,
                    cbar_kws={'label': 'Self-Information'},
                    square=True)
    else:
        sns.heatmap(data,
                    cmap=cmap,
                    annot=False,
                    cbar_kws={'label': 'Self-Information'},
                    square=True)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Width (16)', fontsize=12)
    plt.ylabel('Height (8)', fontsize=12)

    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"热图已保存: {save_path}")


def plot_individual_samples(self_info_data, output_dir, base_name, num_samples=5):
    """
    绘制单个样本的热图
    """
    samples_dir = os.path.join(output_dir, 'individual_samples', base_name)
    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    for i in range(min(num_samples, len(self_info_data))):
        sample_data = self_info_data[i]  # [8, 16]

        plt.figure(figsize=(8, 6))
        sns.heatmap(sample_data,
                    cmap='viridis',
                    annot=False,
                    cbar_kws={'label': 'Self-Information'},
                    square=True)

        plt.title(f'{base_name} - Sample {i + 1}', fontsize=14)
        plt.xlabel('Width')
        plt.ylabel('Height')

        save_path = os.path.join(samples_dir, f'sample_{i + 1}.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"  已保存 {num_samples} 个样本热图到: {samples_dir}")


def visualize_self_info_heatmaps(input_dir, output_dir, file_patterns):
    """
    主函数：可视化自信息热图
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 处理每个文件
    all_mean_data = {}

    for pattern in file_patterns:
        file_path = os.path.join(input_dir, pattern)

        if not os.path.exists(file_path):
            print(f"警告: 文件不存在: {file_path}")
            continue

        print(f"\n处理文件: {pattern}")

        try:
            # 加载自信息数据
            self_info_data = np.load(file_path)  # [10000, 8, 16]
            print(f"  数据形状: {self_info_data.shape}")

            # 计算统计信息
            mean_data = np.mean(self_info_data, axis=0)  # [8, 16]
            std_data = np.std(self_info_data, axis=0)  # [8, 16]
            max_data = np.max(self_info_data, axis=0)  # [8, 16]
            min_data = np.min(self_info_data, axis=0)  # [8, 16]

            all_mean_data[pattern] = mean_data

            print(f"  自信息范围: [{np.min(self_info_data):.4f}, {np.max(self_info_data):.4f}]")
            print(f"  平均自信息: {np.mean(mean_data):.4f} ± {np.mean(std_data):.4f}")

            # 创建基础名称
            base_name = os.path.splitext(pattern)[0]

            # 1. 绘制平均值热图
            create_heatmap(mean_data,
                           f'Mean Self-Information\n{base_name}',
                           os.path.join(output_dir, f'{base_name}_mean.png'))

            # 2. 绘制标准差热图
            create_heatmap(std_data,
                           f'Self-Information Standard Deviation\n{base_name}',
                           os.path.join(output_dir, f'{base_name}_std.png'),
                           cmap='plasma')

            # 3. 绘制最大值热图
            create_heatmap(max_data,
                           f'Maximum Self-Information\n{base_name}',
                           os.path.join(output_dir, f'{base_name}_max.png'))

            # 4. 绘制最小值热图
            create_heatmap(min_data,
                           f'Minimum Self-Information\n{base_name}',
                           os.path.join(output_dir, f'{base_name}_min.png'))

            # 5. 绘制单个样本热图
            plot_individual_samples(self_info_data, output_dir, base_name)

        except Exception as e:
            print(f"处理文件 {pattern} 时出错: {e}")
            continue

    # 如果所有文件都处理成功，创建比较热图
    if len(all_mean_data) > 1:
        print("\n创建比较热图...")
        create_comparison_heatmaps(all_mean_data, output_dir)


def create_comparison_heatmaps(mean_data_dict, output_dir):
    """
    创建比较热图
    """
    # 确定统一的范围
    all_values = np.concatenate([data.flatten() for data in mean_data_dict.values()])
    vmin, vmax = np.min(all_values), np.max(all_values)

    # 创建子图比较
    n_files = len(mean_data_dict)
    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 5))

    if n_files == 1:
        axes = [axes]

    for ax, (file_name, mean_data) in zip(axes, mean_data_dict.items()):
        base_name = os.path.splitext(file_name)[0]
        im = ax.imshow(mean_data, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'{base_name}\nMean', fontsize=12)
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

        # 添加颜色条
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'comparison_mean_heatmaps.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"比较热图已保存: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description='自信息热图可视化')
    parser.add_argument('--input_dir', type=str,
                        default='/data/liuym/LoS_dataset/precomputed_self_info',
                        help='输入自信息npy文件目录')
    parser.add_argument('--output_dir', type=str,
                        default='/data/liuym/LoS_dataset/self_info_heatmaps',
                        help='输出热图目录')
    parser.add_argument('--file_patterns', type=str, nargs='+',
                        default=['self_info_no_patch_LoS_in_test.npy',
                                 'self_info_no_patch_LoS_out_test.npy',
                                 'self_info_no_patch_NLoS_in_test.npy',
                                 'self_info_no_patch_NLoS_out_test.npy'],
                        help='要可视化的自信息文件列表')

    args = parser.parse_args()

    print("开始自信息热图可视化...")
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"处理文件: {args.file_patterns}")

    # 确认操作
    confirm = input(f"确认开始可视化? (y/n): ")
    if confirm.lower() != 'y':
        print("操作已取消")
        return

    visualize_self_info_heatmaps(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_patterns=args.file_patterns
    )

    print(f"\n所有热图可视化完成! 结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()