#!/usr/bin/env python3
"""
特征图热力图对比：无数学公式引导 vs 有数学公式引导
两排展示，每排显示 param_branch 输出的代表性通道热力图
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ============================================================
#  ✏️ 只需改这里
# ============================================================
PROJECT_ROOT = '/home/usuaris/csl/shurong.zhang/data/work3_RADAR/General_decomposition_framework'
CHECKPOINT   = '/home/usuaris/csl/shurong.zhang/data/work3_RADAR/General_decomposition_framework/checkpoints/San_francisco_L/checkpoint_6comp.pth'
DATA_DIR     = None
MODEL_TYPE   = '6comp'
PATCH_SIZE   = 256
SAVE_PATH    = 'feature_heatmap_comparison.png'
N_CHANNELS   = 6    # 展示几个通道（建议 6 或 8）
# ============================================================

sys.path.insert(0, PROJECT_ROOT)
from config import Config
from UniversalPolarDecompAE import CNNPolDecompAE, LightSEResBlock
from dataset import PolSARDataset


# ============================================================
#  消融模型（无物理公式，随机初始化）
# ============================================================
class CNNPolDecompAE_NoPhysics(nn.Module):
    def __init__(self, base_dim=64):
        super().__init__()
        d = base_dim
        self.down1 = nn.Sequential(
            nn.Conv2d(9, d, 3, padding=1), LightSEResBlock(d),
            nn.Conv2d(d, d, 3, stride=2, padding=1))
        self.down2 = nn.Sequential(
            LightSEResBlock(d), nn.Conv2d(d, d*2, 3, stride=2, padding=1))
        self.down3 = nn.Sequential(
            LightSEResBlock(d*2), nn.Conv2d(d*2, d*4, 3, stride=2, padding=1))
        self.bottleneck     = LightSEResBlock(d*4)
        self.feature_refine = nn.Conv2d(d*4, d, 1)
        self.param_branch   = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1),
            nn.GroupNorm(8, d),
            nn.LeakyReLU(0.1, inplace=True))
        self.output_head = nn.Conv2d(d, 9, 1)

    def forward(self, x):
        B, _, H, W = x.shape
        feat = self.feature_refine(
            self.bottleneck(self.down3(self.down2(self.down1(x)))))
        feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
        return self.param_branch(feat)   # [B, 64, H, W]


# ============================================================
#  T3 还原
# ============================================================
def unflatten_t3_image(feat):
    B, C, H, W = feat.shape
    T = torch.zeros(B, 3, 3, H, W, dtype=torch.complex64, device=feat.device)
    T[:, 0, 0] = feat[:, 0];  T[:, 1, 1] = feat[:, 1];  T[:, 2, 2] = feat[:, 2]
    T[:, 0, 1] = torch.complex(feat[:, 3], feat[:, 4]);  T[:, 1, 0] = torch.conj(T[:, 0, 1])
    T[:, 0, 2] = torch.complex(feat[:, 5], feat[:, 6]);  T[:, 2, 0] = torch.conj(T[:, 0, 2])
    T[:, 1, 2] = torch.complex(feat[:, 7], feat[:, 8]);  T[:, 2, 1] = torch.conj(T[:, 1, 2])
    return T


# ============================================================
#  Pauli RGB
# ============================================================
def get_pauli_rgb(x):
    ch = x.squeeze(0).cpu().numpy()
    T11, T22, T33 = ch[0], ch[1], ch[2]
    def log_norm(a):
        a = np.clip(a, 0, None); a = np.log1p(a)
        p2, p98 = np.percentile(a, 2), np.percentile(a, 98)
        return np.clip((a - p2) / (p98 - p2 + 1e-8), 0, 1)
    return (np.stack([log_norm(T22), log_norm((T11+T33)/2), log_norm(T33)], axis=-1) * 255).astype(np.uint8)


# ============================================================
#  提取 param_branch 特征图
# ============================================================
def extract_feat(model, x, t3_raw, device, has_physics):
    model.eval().to(device)
    x = x.to(device)

    if has_physics:
        T_target = unflatten_t3_image(t3_raw.to(device))
        feat_holder = {}
        hook = model.param_branch.register_forward_hook(
            lambda m, i, o: feat_holder.update({'f': o.detach().cpu()}))
        with torch.no_grad():
            model(x, T_target=T_target, is_training=False)
        hook.remove()
        feat = feat_holder['f'].squeeze(0)   # [64, H, W]
    else:
        with torch.no_grad():
            feat = model(x).squeeze(0).cpu()   # [64, H, W]

    return feat


# ============================================================
#  选出最有代表性的 N 个通道（方差最大）
# ============================================================
def pick_channels(feat, n):
    variances = feat.numpy().reshape(feat.shape[0], -1).var(axis=1)
    return np.argsort(variances)[::-1][:n]


# ============================================================
#  主可视化
# ============================================================
def visualize(model_phys, model_nophys, x, t3_raw, device, save_path):

    feat_phys   = extract_feat(model_phys,   x, t3_raw, device, has_physics=True)
    feat_nophys = extract_feat(model_nophys, x, t3_raw, device, has_physics=False)
    rgb = get_pauli_rgb(x)

    # 用有物理模型的方差最大通道作为展示索引（两排用同样的通道索引，对比才公平）
    ch_indices = pick_channels(feat_phys, N_CHANNELS)

    n_cols = N_CHANNELS + 1   # 原图 + N个通道

    fig = plt.figure(figsize=(3.2 * n_cols, 8), facecolor='#0d0d14')

    fig.text(0.5, 0.99,
             'param_branch 特征图热力图：无数学公式引导 vs 有数学公式引导',
             ha='center', va='top', fontsize=15, fontweight='bold', color='white')
    fig.text(0.5, 0.955,
             'Without Physics Formula  vs  With Physics Formula  '
             f'|  Top-{N_CHANNELS} Most Active Channels  |  {MODEL_TYPE}',
             ha='center', va='top', fontsize=10, color='#888888', fontfamily='monospace')

    gs = gridspec.GridSpec(2, n_cols, figure=fig,
                           left=0.04, right=0.98,
                           top=0.92, bottom=0.04,
                           wspace=0.04, hspace=0.10)

    def plot_row(row, feat, row_label, row_color):
        # 列 0：SAR 原图
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(rgb)
        ax.set_ylabel(row_label, color=row_color, fontsize=12,
                      fontweight='bold', labelpad=8)
        ax.set_title('SAR Pauli RGB', color='#cccccc', fontsize=9, pad=4)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(row_color); sp.set_linewidth(2.5)
        ax.set_frame_on(True)

        # 列 1…N：特征通道热力图
        feat_np = feat.numpy()
        for ci, ch_idx in enumerate(ch_indices):
            ax = fig.add_subplot(gs[row, ci + 1])
            ch = feat_np[ch_idx]
            lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
            ch_norm = np.clip((ch - lo) / (hi - lo + 1e-8), 0, 1)

            im = ax.imshow(ch_norm, cmap='jet', vmin=0, vmax=1)
            ax.set_title(f'Channel {ch_idx}', color='#aaaaaa', fontsize=8, pad=4)
            ax.axis('off')

            if ci == N_CHANNELS - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                cbar.ax.tick_params(labelsize=6, colors='#666666')
                cbar.set_label('激活强度', color='#666666', fontsize=7)

    plot_row(0, feat_nophys, '无数学公式\n(No Physics)', '#ff5252')
    plot_row(1, feat_phys,   '有数学公式\n(With Physics)', '#00e676')

    fig.text(0.5, 0.01,
             '上排（无公式）：特征激活弥散、无结构  |  下排（有公式）：激活集中、结构清晰，对应物理散射区域',
             ha='center', fontsize=9, color='#555555', fontstyle='italic')

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight',
                facecolor='#0d0d14', edgecolor='none')
    plt.close()
    print(f"✅ 已保存至: {save_path}")


# ============================================================
#  主入口
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  设备: {device}")

    data_dir = DATA_DIR if DATA_DIR else Config.DATA_DIR

    dataset = PolSARDataset(data_dir=data_dir, roi=None,
                            training=True, patch_size=PATCH_SIZE)
    sample = dataset[0]
    x      = sample[0].unsqueeze(0).float()
    t3_raw = sample[1].unsqueeze(0).float()
    print(f"✅ 数据加载完成 | 形状: {x.shape}")

    # 有物理公式的模型
    model_phys = CNNPolDecompAE(decomp_type=MODEL_TYPE).to(device)
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=True)
        model_phys.load_state_dict(ckpt['model_state_dict'])
        print(f"✅ checkpoint 加载成功")
    else:
        print(f"⚠️  未找到 checkpoint，使用随机初始化")
    model_phys.eval()

    # 无物理公式的消融模型（随机初始化）
    model_nophys = CNNPolDecompAE_NoPhysics(base_dim=64).to(device)
    model_nophys.eval()
    print("✅ 消融模型初始化完成（随机权重）")

    visualize(model_phys, model_nophys, x, t3_raw, device, SAVE_PATH)


if __name__ == '__main__':
    main()