#!/usr/bin/env python3


import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from config import Config


def parse_hdr_dimensions(hdr_path: str) -> Tuple[int, int]:

    if not os.path.exists(hdr_path):
        raise FileNotFoundError(f"HDR file not found: {hdr_path}")
    
    with open(hdr_path, 'r') as f:
        content = f.read()
    
    lines_content = [line.strip() for line in content.splitlines()]
    samples_str = next((line.split('=')[1].strip() for line in lines_content if 'samples' in line.lower()), None)
    lines_str = next((line.split('=')[1].strip() for line in lines_content if 'lines' in line.lower()), None)
    
    if samples_str is None or lines_str is None:
        raise ValueError(f"Missing 'samples' or 'lines' in {hdr_path}")
    
    return int(lines_str), int(samples_str)  # H, W


def load_channel(path: str, shape: Tuple[int, int], roi: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Binary file not found: {path}")
    
    H, W = shape
    
    if roi is None:
        # 加载完整图像
        data = np.fromfile(path, dtype=np.float32)
        if len(data) != H * W:
            raise ValueError(f"Data length mismatch: {len(data)} vs {H * W} in {path}")
        return data.reshape(H, W)
    else:
        # 加载ROI区域
        row_start, row_end, col_start, col_end = roi
        data = np.fromfile(path, dtype=np.float32).reshape(H, W)
        return data[row_start:row_end, col_start:col_end]


def load_polsar_channels(
    data_dir: str = None, 
    roi: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[np.ndarray, Tuple[int, int]]:

    if data_dir is None:
        data_dir = Config.DATA_DIR
 
    hdr_path = os.path.join(data_dir, Config.HDR_FILE)
    H_full, W_full = parse_hdr_dimensions(hdr_path)
    print(f"完整图像尺寸: {H_full} x {W_full}")
    
    if roi is not None:
        row_start, row_end, col_start, col_end = roi
        
        if not (0 <= row_start < row_end <= H_full and 0 <= col_start < col_end <= W_full):
            raise ValueError(f"Invalid ROI [{row_start}:{row_end}, {col_start}:{col_end}] for image size {H_full}x{W_full}")
        
        H_roi = row_end - row_start
        W_roi = col_end - col_start
        print(f"ROI区域: 行[{row_start}:{row_end}], 列[{col_start}:{col_end}]")
        print(f"ROI尺寸: {H_roi} x {W_roi}")
    else:
        print("使用完整图像")

    channel_files = [
        'T11.bin', 'T22.bin', 'T33.bin',
        'T12_real.bin', 'T12_imag.bin',
        'T13_real.bin', 'T13_imag.bin',
        'T23_real.bin', 'T23_imag.bin'
    ]
    channel_names = [
        'T11', 'T22', 'T33',
        'T12_real', 'T12_imag',
        'T13_real', 'T13_imag',
        'T23_real', 'T23_imag'
    ]

    feat_channels = []
    for fname, cname in zip(channel_files, channel_names):
        path = os.path.join(data_dir, fname)
        channel = load_channel(path, (H_full, W_full), roi)
        print(f"加载 {cname}: shape {channel.shape}, range [{channel.min():.4f}, {channel.max():.4f}]")
        feat_channels.append(channel)

    feat = np.stack(feat_channels, axis=-1)  # [H, W, 9]
    actual_shape = feat.shape[:2]
    return feat, actual_shape


def global_outlier_removal(
    feat: np.ndarray,
    denoise_window: int = None,  
    clip_low_pct: int = None,
    clip_high_pct: int = None
) -> np.ndarray:

    if denoise_window is None:
        denoise_window = Config.DENOISE_WINDOW
    if clip_low_pct is None:
        clip_low_pct = Config.CLIP_LOW_PCT
    if clip_high_pct is None:
        clip_high_pct = Config.CLIP_HIGH_PCT
    
    H, W, C = feat.shape
    processed_feat = feat.copy()
    kernel = np.ones((denoise_window, denoise_window), dtype=np.float32) / (denoise_window ** 2)

    for c in range(C):
        channel_data = processed_feat[..., c]

        valid_data = channel_data[~np.isnan(channel_data)]
        if len(valid_data) == 0:
            print(f"Warning: Channel {c} is all NaN; skipping.")
            continue

        low_thresh = np.percentile(valid_data, clip_low_pct)
        high_thresh = np.percentile(valid_data, clip_high_pct)

        is_outlier = (channel_data < low_thresh) | (channel_data > high_thresh) | np.isnan(channel_data)
        if not np.any(is_outlier):
            continue

        normal_mask = np.where(is_outlier, 0.0, 1.0).astype(np.float32)
        valid_count = cv2.filter2D(normal_mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        normal_data = np.where(is_outlier, 0.0, channel_data)
        neighbor_sum = cv2.filter2D(normal_data, -1, kernel, borderType=cv2.BORDER_REFLECT)

        channel_median = np.median(valid_data)
        neighbor_mean = np.divide(
            neighbor_sum,
            valid_count,
            out=np.full_like(neighbor_sum, channel_median),
            where=(valid_count > 0)
        )

        channel_data[is_outlier] = neighbor_mean[is_outlier]
        processed_feat[..., c] = channel_data

    return processed_feat



def assemble_t_matrix(
    feat_processed: np.ndarray,
    shape: Tuple[int, int]
) -> torch.Tensor:
    
    H, W = shape
    T11 = feat_processed[..., 0]
    T22 = feat_processed[..., 1]
    T33 = feat_processed[..., 2]
    T12_real, T12_imag = feat_processed[..., 3], feat_processed[..., 4]
    T13_real, T13_imag = feat_processed[..., 5], feat_processed[..., 6]
    T23_real, T23_imag = feat_processed[..., 7], feat_processed[..., 8]

    T = np.zeros((H, W, 3, 3), dtype=np.complex64)
    T[..., 0, 0] = T11
    T[..., 1, 1] = T22
    T[..., 2, 2] = T33
    T[..., 0, 1] = T12_real + 1j * T12_imag
    T[..., 1, 0] = np.conj(T[..., 0, 1])
    T[..., 0, 2] = T13_real + 1j * T13_imag
    T[..., 2, 0] = np.conj(T[..., 0, 2])
    T[..., 1, 2] = T23_real + 1j * T23_imag
    T[..., 2, 1] = np.conj(T[..., 1, 2])

    return torch.from_numpy(T)


def visualize_channel(
    data: np.ndarray,
    filename: str,
    cmap: str = 'gray',
    dpi: int = 300
) -> None:
   
    vmin, vmax = 0, 0.5
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved visualization: {filename}")

DATA_DIR = Config.DATA_DIR
HDR_FILE = Config.HDR_FILE