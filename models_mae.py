# models_mae_jepa.py
# --------------------------------------------------------
# MAE + JEPA-style latent prediction + SIGReg (lightweight)
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


# --------------------------------------------------------
# 1. SIGReg (lightweight, subsampled)
# --------------------------------------------------------

class SlicedIsotropicGaussianRegularization(nn.Module):
    def __init__(self, num_slices=128, num_points=17, domain=(-5.0, 5.0)):
        super().__init__()
        self.num_slices = num_slices
        self.num_points = num_points
        self.domain = domain

    def forward(self, embeddings):
        # embeddings: [N, D]
        device = embeddings.device
        N, D = embeddings.shape

        A = torch.randn(self.num_slices, D, device=device)
        A = A / (A.norm(dim=1, keepdim=True) + 1e-6)

        proj = embeddings @ A.T  # [N, S]

        t = torch.linspace(self.domain[0], self.domain[1],
                           self.num_points, device=device)
        dt = t[1] - t[0]

        phi_t = torch.exp(-0.5 * t ** 2)  # Gaussian CF

        t = t[:, None, None]              # [T,1,1]
        u = proj[None, :, :]              # [1,N,S]

        exp_itu = torch.exp(1j * t * u)
        phi_u = exp_itu.mean(dim=1)       # [T,S]

        diff = phi_u - phi_t[:, None]
        sq = diff.real ** 2 + diff.imag ** 2

        integrals = dt * (
            sq[0] / 2 + sq[-1] / 2 + sq[1:-1].sum(dim=0)
        )

        return integrals.mean()


# --------------------------------------------------------
# 2. MAE + JEPA model
# --------------------------------------------------------

class MAE_JEPA_ViT(nn.Module):

    def __init__(self,
                 img_size=128,
                 patch_size=2,
                 in_chans=16,
                 embed_dim=192,
                 depth=12,
                 num_heads=6,
                 decoder_embed_dim=64,
                 decoder_depth=4,
                 decoder_num_heads=4,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False,
                 use_checkpoint=False,
                 sigreg_lambda=0.02,
                 jepa_lambda=0.2,
                 freeze_decoder_layers=0):

        super().__init__()

        # ----------- 必须最先保存超参数（可选，但推荐） -----------
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_chans = in_chans

        # ----------- 必须先创建 patch_embed -----------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches  # 这里才能访问

        # 可选：保存 num_patches 供后续使用（很多地方会用到）
        self.num_patches = num_patches

        # ----------- 其他属性 -----------
        self.norm_pix_loss = norm_pix_loss
        self.use_checkpoint = use_checkpoint
        self.sigreg_lambda = sigreg_lambda
        self.jepa_lambda = jepa_lambda

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # -------- JEPA latent predictor --------
        # -------- JEPA latent predictor (修改：从MLP改为小型Transformer，能预测masked latents) --------
        # 原：nn.Sequential(Linear, GELU, Linear) – 无法改变序列长度
        # 新：用2层Block，输入visible + masked_placeholder，输出masked部分
        self.latent_pred_embed = nn.Linear(embed_dim, embed_dim)  # 投影visible
        self.latent_pred_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # placeholder for masked
        self.latent_pred_pos_embed = nn.Parameter(  # 共享encoder的pos_embed，但requires_grad=True
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True
        )
        self.latent_pred_blocks = nn.ModuleList([
            Block(embed_dim, num_heads=3, mlp_ratio=2., qkv_bias=True, norm_layer=norm_layer)
            # 轻量：heads=3, mlp=2, depth=2
            for _ in range(2)  # 浅层以节省内存
        ])
        self.latent_pred_norm = norm_layer(embed_dim)
        # 输出头：投影回embed_dim，仅用于masked部分
        self.latent_pred_head = nn.Linear(embed_dim, embed_dim)

        # -------- Decoder (MAE) --------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            patch_size ** 2 * in_chans
        )

        # -------- SIGReg --------
        self.sigreg = SlicedIsotropicGaussianRegularization()

        self.initialize_weights()

    # --------------------------------------------------------

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        dec_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(dec_pos_embed).float().unsqueeze(0)
        )

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0)

    def get_masked_ids(self, mask, ids_restore):
        B, L = mask.shape
        device = mask.device
        full_arange = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)
        masked_ids = full_arange[mask == 1].reshape(B, -1)
        return masked_ids

        return masked_ids  # [B, L_masked]
    # --------------------------------------------------------
    # MAE utilities
    # --------------------------------------------------------

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        return x.reshape(B, h * w, p * p * C)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_masked, mask, ids_restore

    # --------------------------------------------------------
    # Encoder / Decoder
    # --------------------------------------------------------

    def forward_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat([cls.expand(x.size(0), -1, -1), x], dim=1)

        for blk in self.blocks:
            x = cp.checkpoint(blk, x) if self.use_checkpoint else blk(x)

        # 新增：计算masked_ids
        masked_ids = self.get_masked_ids(mask, ids_restore)

        return self.norm(x), mask, ids_restore, masked_ids  # 多返回masked_ids

    def forward_decoder(self, latent, ids_restore, pred_latent):
        # pred_latent: predicted_masked_latents [B, L_masked, embed_dim]

        x = self.decoder_embed(latent)  # [B, L_visible + 1, decoder_embed_dim]

        # 修改：用pred_latent替换mask_tokens
        # 原：mask_tokens = self.mask_token.repeat(...)  # [B, L_masked, decoder_embed_dim]
        # 新：pred_latent已传入，但需投影到decoder_dim（如果embed_dim != decoder_embed_dim）
        pred_latent = self.decoder_embed(pred_latent)

        # 拼接：可见patches + predicted_masked
        x_ = torch.cat([x[:, 1:, :], pred_latent],
                       dim=1)  # [B, L_visible + L_masked, decoder_embed_dim] = [B, num_patches, ...]

        # unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # 添加cls
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        if self.use_checkpoint:
            for blk in self.decoder_blocks:
                x = cp.checkpoint(blk, x)
        else:
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        return x

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        denominator = (target ** 2).mean(dim=-1)
        denominator = (denominator * mask).sum() / mask.sum()
        NMSE = loss / (denominator + 1e-6)

        return loss, NMSE, pred  # 必须返回三个值

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(self, imgs, self_infos=None, mask_ratio=0.75, target=None):
        # ------------------- Masked view encoder -------------------
        latent, mask, ids_restore, masked_ids = self.forward_encoder(imgs, mask_ratio)

        B, num_patches = mask.shape
        L_masked = masked_ids.shape[1]
        L_visible = num_patches - L_masked

        # ------------------- Full view encoder (for JEPA target & SIGReg) -------------------
        full_latent, _, _, _ = self.forward_encoder(imgs, 0.0)
        z_full = full_latent[:, 1:, :]  # [B, num_patches, D]

        # ------------------- JEPA latent prediction -------------------
        z_visible = latent[:, 1:, :]  # [B, L_visible, D]

        # 创建 masked placeholder
        masked_placeholder = self.latent_pred_mask_token.repeat(B, L_masked, 1)
        masked_pos = self.latent_pred_pos_embed[:, 1:, :][0, masked_ids, :]  # [B, L_masked, D]
        masked_placeholder = masked_placeholder + masked_pos

        visible_embedded = self.latent_pred_embed(z_visible)

        pred_input = torch.cat([visible_embedded, masked_placeholder], dim=1)  # [B, num_patches, D]

        # 恢复原始空间顺序（可选但推荐）
        visible_ids = torch.where(mask == 0)[1].reshape(B, L_visible)
        full_ids = torch.cat([visible_ids, masked_ids], dim=1)
        ids_restore_pred = torch.argsort(full_ids, dim=1)
        pred_input = torch.gather(pred_input, 1, ids_restore_pred.unsqueeze(-1).repeat(1, 1, self.embed_dim))

        # predictor blocks
        for blk in self.latent_pred_blocks:
            pred_input = blk(pred_input)
        pred_input = self.latent_pred_norm(pred_input)

        # 提取 masked 部分的预测
        predicted_masked_latents = torch.gather(
            pred_input, 1, masked_ids.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        )
        predicted_masked_latents = self.latent_pred_head(predicted_masked_latents)  # [B, L_masked, D]

        # JEPA loss
        true_masked_latents = torch.gather(
            z_full, 1, masked_ids.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        )
        jepa_loss = F.mse_loss(predicted_masked_latents, true_masked_latents)

        # ------------------- Decoder with predicted latents -------------------
        pred_pixels = self.forward_decoder(latent, ids_restore, predicted_masked_latents)

        # ------------------- MAE reconstruction loss + NMSE -------------------
        # 注意：这里调用 forward_loss，返回 loss, NMSE, pred
        mae_loss, NMSE, _ = self.forward_loss(imgs, pred_pixels, mask)
        # 原 forward_loss 返回 (loss, NMSE, pred)，我们只需要前两个

        # ------------------- SIGReg -------------------
        z_flat = z_full.reshape(-1, self.embed_dim)
        idx = torch.randperm(z_flat.shape[0], device=z_flat.device)[:2048]
        sigreg_loss = self.sigreg(z_flat[idx])

        # ------------------- Total loss (对外只暴露 mae_loss 作为 NMSE 的依据) -------------------
        total_loss = mae_loss + self.jepa_lambda * jepa_loss + self.sigreg_lambda * sigreg_loss

        # ------------------- 兼容原接口：返回 (loss, NMSE, pred, mask) -------------------
        return total_loss, NMSE, pred_pixels, mask


# --------------------------------------------------------
# Factory
# --------------------------------------------------------

def mae_jepa_vit_base_patch2(**kwargs):
    return MAE_JEPA_ViT(
        patch_size=2,
        embed_dim=192,
        depth=12,
        num_heads=6,
        decoder_embed_dim=kwargs.pop('decoder_embed_dim', 64),
        decoder_depth=kwargs.pop('decoder_depth', 4),
        decoder_num_heads=kwargs.pop('decoder_num_heads', 4),
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint=kwargs.pop('use_checkpoint', False),
        freeze_decoder_layers=kwargs.pop('freeze_decoder_layers', 0),
        **kwargs
    )
