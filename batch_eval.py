import subprocess

# 定义要测试的 mask_ratio 值
mask_ratios = [0.99,0.95,0.9]
# mask_ratios = [  0.85, 0.8, 0.75, 0.7, 0.5,0.4,0.3,0.2,0.1]
# noise_dbs = [None]  # None表示无噪声情况
# noise_dbs = [None, 20 ,15 ,10,5, 0]  # None表示无噪声情况
noise_dbs = [10,5, 0]  # None表示无噪声情况


# 遍历每个 mask_ratio 并执行 model_eval.py
for noise in noise_dbs:
    for ratio  in mask_ratios:
        # 构建基本命令
        command = (
            f"CUDA_VISIBLE_DEVICES=1 python model_eval_huabu.py "
            f"--model mae_vit_base_patch2 "
            f"--mask_ratio {ratio} "
            f"--batch_idx 0 "
            f"--sample_index 0 "
            f"--norm_pix_loss "
            f"--batch_size 8 " 
            f"--input_size 128"
        )
        
        # 如果有噪声参数，则添加到命令中
        if noise is not None:
            command += f" --noise_db {noise}db"
        
        print(f"Running command: {command}")
        subprocess.run(command, shell=True)

#CUDA_VISIBLE_DEVICES=1 
