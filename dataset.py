#!/usr/bin/env python3
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from data_import import (
    load_polsar_channels, global_outlier_removal, DATA_DIR
)


class PolSARDataset(Dataset):

    def __init__(
        self,
        data_dir: str = DATA_DIR,
        denoise_window: int = 10,
        clip_low_pct: int = 1,
        clip_high_pct: int = 99,
        training: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None,
        patch_size: int = 128,
        samples_per_epoch: int = 1000
    ):
        self.data_dir = data_dir
        self.training = training
        self.patch_size = patch_size
        self.roi = roi
        self.samples_per_epoch = samples_per_epoch
        self.EPS = 1e-8  # 统一EPS常量，避免分散定义

        print(f"\n{'='*70}")
        print(f"初始化PolSARDataset | 模式: {'训练' if training else '推理'}")
        print(f"{'='*70}")
        
        # 1. 加载原始PolSAR数据
        feat, shape = load_polsar_channels(data_dir, roi=self.roi)
        self.H, self.W = shape
        print(f"✅ 数据加载完成 | 尺寸: {self.H} x {self.W}")

        # 2. 异常值去除
        feat_processed = global_outlier_removal(
            feat, denoise_window, clip_low_pct, clip_high_pct
        )
        print(f"✅ 异常值去除完成")

        EPS = 1e-8
        feat_log = np.zeros_like(feat_processed)
        # 对角线
        for i in [0, 1, 2]: 
            feat_log[..., i] = 10 * np.log10(np.maximum(feat_processed[..., i], EPS))
        # 非对角线
        for i in range(3, 9, 2):  # (3,4), (5,6), (7,8)
            real = feat_processed[..., i]
            imag = feat_processed[..., i+1]
    
            magnitude = np.sqrt(real**2 + imag**2) + EPS
            phase = np.arctan2(imag, real)
     
            mag_db = 10 * np.log10(magnitude)
    
            feat_log[..., i] = mag_db * np.cos(phase)
            feat_log[..., i+1] = mag_db * np.sin(phase)
        
        # 计算统计量 (基于当前加载的 ROI 数据)
        flat_log = feat_log.reshape(-1, 9)
        self.mean = np.mean(flat_log, axis=0)
        self.std = np.std(flat_log, axis=0) + 1e-6
        
        print(f"📊 归一化统计:")
        print(f"   强度通道(dB) - Mean: {self.mean[:3]}")
        print(f"   强度通道(dB) - Std:  {self.std[:3]}")

        # 转为 Tensor: [H, W, 9] -> [9, H, W] (符合 PyTorch CNN 格式)
        self.input_full = torch.from_numpy(feat_log).permute(2, 0, 1).float()
        self.target_full = torch.from_numpy(feat_processed).permute(2, 0, 1).float()

        print(f"✅ 数据预处理完成")
        print(f"   输入形状: {self.input_full.shape}")
        print(f"   目标形状: {self.target_full.shape}")
        print(f"{'='*70}\n")


    def __len__(self) -> int:
        return self.samples_per_epoch if self.training else 1
    
    def __getitem__(self, idx: int):
        # 准备归一化参数：转为[9,1,1]，支持广播到[9, H, W]
        mean = torch.from_numpy(self.mean).view(9, 1, 1).float()
        std = torch.from_numpy(self.std).view(9, 1, 1).float()

        # 推理模式：返回整张ROI的归一化输入、原始目标、坐标(0,0)
        if not self.training:
            norm_input = (self.input_full - mean) / std
            return norm_input, self.target_full, (0, 0)

        # 训练模式：随机切Patch（保证不越界）
        h_start = np.random.randint(0, self.H - self.patch_size + 1)
        w_start = np.random.randint(0, self.W - self.patch_size + 1)
        
        # 切片获取Patch
        img_crop = self.input_full[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
        target_crop = self.target_full[:, h_start:h_start+self.patch_size, w_start:w_start+self.patch_size]
        
        # 仅对输入Patch做归一化
        norm_crop = (img_crop - mean) / std
        
        # 返回：归一化输入、对数域目标、Patch起始坐标
        return norm_crop, target_crop, (h_start, w_start)


def polsar_collate_fn(batch):
    """
    自定义collate函数，正确处理batch中的坐标信息
    Args:
        batch: list of (input, target, coords)，每个元素为__getitem__的返回值
    Returns:
        batch_input: torch.Tensor [B, C, H, W]，批量输入
        batch_target: torch.Tensor [B, C, H, W]，批量目标
        batch_coords: list of (h, w)，批量Patch起始坐标
    """
    inputs = []
    targets = []
    coords = []
    
    for item in batch:
        if len(item) == 3:
            inp, tar, coord = item
            coords.append(coord)
        else:
            inp, tar = item
            coords.append((0, 0))
        inputs.append(inp)
        targets.append(tar)
    
    # 堆叠为batch维度
    batch_input = torch.stack(inputs, dim=0)
    batch_target = torch.stack(targets, dim=0)
    
    return batch_input, batch_target, coords