# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from timm.models.vit_moe import PatchEmbed, BlockWithMoE  # 从自定义模块导入

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=64, patch_size=16, in_chans=16,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_moe=True,  # 新增参数：是否启用 MoE
                 num_experts=4,  # 新增参数：专家数量
                 top_k=2,  # 新增参数：激活的专家数
                 qkv_bias = True,  # 添加 qkv_bias 参数
                 drop_rate = 0.0,  # 添加 drop_rate 参数
                 attn_drop_rate = 0.0,  # 添加 attn_drop_rate 参数
                 drop_path_rate = 0.0,  # 添加 drop_path_rate 参数
                 qk_scale = None  # 添加 qk_scale 参数（可选）
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            BlockWithMoE(
                dim=embed_dim,
                num_heads=num_heads,
                num_experts=num_experts,
                top_k=top_k,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,  # 使用传入的 qkv_bias
                qk_scale=qk_scale,  # 使用传入的 qk_scale
                drop=drop_rate,  # 使用传入的 drop_rate
                attn_drop=attn_drop_rate,  # 使用传入的 attn_drop_rate
                drop_path=drop_path_rate,  # 使用传入的 drop_path_rate
                norm_layer=norm_layer
            )for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        self.use_clean_target = False  # 默认使用原始 forward 方法

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def generate_mask_with_self_information(self, self_information, observe_ratio):
        """
        根据自信息生成掩码 (保留高自信息区域)

        Args:
            self_information: 形状为 (Ny, Nx) 的自信息
            observe_ratio: 观察比例，即保留区域的比例

        Returns:
            mask: 形状为 (Ny, Nx) 的掩码，0 表示掩码，1 表示保留
        """
        Ny, Nx = self_information.shape
        num_observe = int(Ny * Nx * observe_ratio)

        # 将自信息转换为一维数组，并获取索引 (降序，保留高自信息)
        # 修改后（低自信息保留）
        flattened_indices = torch.argsort(self_information.flatten(), descending=False)  # 改为升序

        # 创建掩码
        mask = torch.zeros((Ny, Nx), dtype=torch.int)
        observe_indices = flattened_indices[:num_observe]  # 选取最大的 self_information
        mask[np.unravel_index(observe_indices.cpu().numpy(), (Ny, Nx))] = 1

        return mask.numpy()

    def add_sparse_variable_noise(self,x, noise_prob=0.05, sigma_range=(0.0, 0.5)):
        """
        向图像添加稀疏的、幅度可变的高斯噪声

        Args:
            x (torch.Tensor): 输入图像张量 (B, C, H, W)
            noise_prob (float): 每个像素点添加噪声的概率 (0.0 ~ 1.0)
            sigma_range (tuple): 噪声标准差的范围 (min_sigma, max_sigma)

        Returns:
            torch.Tensor: 添加噪声后的图像张量 (B, C, H, W)
        """

        B, C, H, W = x.shape

        # 1. 生成随机概率掩码，决定哪些像素添加噪声
        noise_mask = torch.rand(B, 1, H, W, device=x.device) < noise_prob  # 注意放在正确的设备上
        noise_mask = noise_mask.expand(B, C, H, W)  # 扩展到所有通道

        # 2. 生成随机噪声强度系数，幅度在 sigma_range 内
        min_sigma, max_sigma = sigma_range
        sigma = min_sigma + (max_sigma - min_sigma) * torch.rand(B, 1, H, W, device=x.device)  # 随机标准差
        sigma = sigma.expand(B, C, H, W)  # 扩展到所有通道

        # 3. 生成高斯噪声
        noise = torch.randn_like(x) * sigma

        # 4. 根据 noise_mask 添加噪声
        noisy_x = x + noise * noise_mask.float()  # 确保 mask 是 float 类型

        return noisy_x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 16, H, W)
        x: (N, L, patch_size**2 *16)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 16, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 16))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *16)
        imgs: (N, 16, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 16))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 16, h * p, h * p))
        return imgs

    def random_masking1(self, x, self_infos, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        # print(f"x shape : {x.shape}")

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # print(f"mask shape : {mask.shape}")
        # print(f"ids_restore shape : {ids_restore.shape}")

        return x_masked, mask, ids_restore

    def random_masking(self, x, self_infos, mask_ratio):
        """
        基于 self_infos 生成掩码，并返回 x_masked, mask, ids_restore
        x: [N, L, D], L = h*w
        self_infos: [N, h, w]
        """
        N, L, D = x.shape
        h = w = int(L ** 0.5)
        # print(f"x shape : {x.shape}")
        mask = torch.zeros((N, L), device=x.device)  # 初始化全0掩码

        # 为每个样本生成 mask_2d 并展平
        for n in range(N):
            # 提取当前样本的 self_information
            self_info = self_infos[n]  # [h, w]

            # 生成二维掩码（保留高自信息区域）
            mask_2d = self.generate_mask_with_self_information(
                self_info,
                observe_ratio=1 - mask_ratio  # 保留比例为 1 - mask_ratio
            )  # [h, w], 1表示保留

            # 将 mask_2d 展平为 [L]
            mask_flat = 1- torch.from_numpy(mask_2d).flatten().to(x.device)

            # 反转逻辑：原函数生成的是保留掩码，此处需转换为MAE的掩码格式（0保留，1丢弃）
            mask[n] = mask_flat  # 0=保留，1=丢弃

        # 计算每个样本保留的索引
        ids_shuffle = []
        for n in range(N):
            # 获取保留和掩码的索引
            keep_indices = torch.where(mask[n] == 0)[0].cpu().numpy()  # 保留的位置（0=保留）
            remove_indices = torch.where(mask[n] == 1)[0].cpu().numpy()  # 掩码的位置（1=掩码）

            # 随机打乱保留的索引（模拟噪声排序）
            np.random.shuffle(keep_indices)  # 引入随机性
            combined_indices = np.concatenate([keep_indices, remove_indices])
            ids_shuffle.append(torch.tensor(combined_indices, device=x.device))

        ids_shuffle = torch.stack(ids_shuffle)  # [N, L]

        # 步骤2：生成 ids_restore（恢复原始顺序的索引）
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # [N, L]

        # 步骤3：收集保留的 tokens（前 len_keep 个位置）
        len_keep = int(L * (1 - mask_ratio))
        ids_keep = ids_shuffle[:, :len_keep]  # [N, len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        # print(f"mask shape : {mask.shape}")
        # print(f"ids_restore shape : {ids_restore.shape}")

        return x_masked, mask, ids_restore

    def random_masking_low(self, x, self_infos, mask_ratio):
        """
        基于概率的掩码策略：自信息值小的patch有更高概率被保留
        x: [N, L, D] 输入序列 (L = h*w)
        self_infos: [N, h, w] 自信息图（值在[0,1]）
        """
        N, L, D = x.shape
        h, w = self_infos.shape[1], self_infos.shape[2]

        # 1. 直接使用自信息计算保留概率（自信息小→高保留概率）
        p_keep = 1.0 - self_infos  # 反转自信息值

        # 2. 添加基础概率确保最小值
        base_prob = 0.01
        p_keep = p_keep * (1 - base_prob) + base_prob

        # 3. 展平概率
        p_keep_flat = p_keep.reshape(N, -1)

        # 4. 确保最小保留patch数
        min_keep = max(1, int(L * 0.05))  # 至少保留5%
        len_keep = max(min_keep, int(L * (1 - mask_ratio)))

        # 5. 多项式采样
        indices = torch.zeros((N, len_keep), dtype=torch.long, device=x.device)
        for i in range(N):
            # 归一化概率
            probs = p_keep_flat[i]
            probs /= probs.sum()

            # 采样
            indices[i] = torch.multinomial(
                probs,
                num_samples=len_keep,
                replacement=False
            )

        # 6. 创建mask
        mask = torch.ones((N, L), device=x.device)
        mask.scatter_(1, indices, 0)  # 0=保留

        # 7. 构建恢复索引
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 8. 收集保留的tokens
        x_masked = torch.gather(
            x,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, self_infos, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking1(x, self_infos, mask_ratio)
        # print(f"x shape after patch_embed: {x.shape}")  # 调试信息

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # print(f"x shape after appending cls_token: {x.shape}")  # 调试信息

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # print(f"x shape after appending blk: {x.shape}")  # 调试信息

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 16, H, W]
        pred: [N, L, p*p*16]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            # print(f"norm_pix_loss is True ")
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # pred = pred * (var + 1.e-6) ** .5 + mean
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        denominator = (target ** 2)
        denominator = denominator.mean(dim=-1)
        denominator = (denominator * mask).sum() / mask.sum()  # Normalized by target

        NMSE = loss / denominator

        # 去归一化 target
        # if self.norm_pix_loss:
        #     pred = pred * (var + 1.e-6) ** .5 + mean

        return loss, NMSE, pred

    def set_use_clean_target(self, use_clean_target):
        """设置是否使用干净数据作为目标"""
        self.use_clean_target = use_clean_target

    def forward(self, imgs, self_infos=None, mask_ratio=0.75, target=None):
        if self.use_clean_target and target is not None:
            # 使用修改后的 forward 方法，传入干净数据作为目标
            return self.forward_with_clean_target(imgs, self_infos, mask_ratio, target)
        else:
            # 使用原始的 forward 方法
            return self.forward_original(imgs, self_infos, mask_ratio)

    def forward_original(self, imgs, self_infos, mask_ratio=0.75):
        # 添加稀疏噪声（保持形状和数据类型）
        target = imgs
        imgs = self.add_sparse_variable_noise(
            x=imgs,
            noise_prob=0.05,  # 可调节参数
            sigma_range=(0.0, 0.5)  # 可调节参数
        ).to(imgs.dtype)  # 确保数据类型一致
        latent, mask, ids_restore = self.forward_encoder(target, self_infos, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, NMSE, pred_gai = self.forward_loss(target, pred, mask)
        return loss, NMSE, pred, mask

    def forward_with_clean_target(self, imgs, self_infos, mask_ratio=0.75, target=None):
        """修改后的 forward 方法，使用干净数据作为目标"""
        latent, mask, ids_restore = self.forward_encoder(imgs, self_infos, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss, NMSE, pred = self.forward_loss(target, pred, mask)  # 使用干净数据计算损失
        return loss, NMSE, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch2_dec192d8b(**kwargs):
    model = MaskedAutoencoderViT(
        use_moe=True,
        num_experts=4,
        top_k=2,
        patch_size=2, embed_dim=192, depth=12, num_heads=6,
        decoder_embed_dim=128, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,        # 显式传递
        qk_scale=None,        # 显式传递
        drop_rate=0.1,        # 显式传递
        attn_drop_rate=0.1,   # 显式传递
        drop_path_rate=0.1,   # 显式传递
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def mae_vit_base_patch16_dec4096d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=4096, depth=12, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dectest(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch2 = mae_vit_base_patch2_dec192d8b
mae_vit_base_patch16_test = mae_vit_base_patch16_dec4096d8b
mae_vit_base_patch16_dectest = mae_vit_base_patch16_dectest
