#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from enum import Enum
import numpy as np
import os


class ModelType(Enum):
    comp3 = '3comp'
    comp4 = '4comp'
    comp6 = '6comp'


EPS = 1e-8

class ReferenceDataLoader:
    """加载参考分解数据"""
    def __init__(self, reference_dir: str, decomp_type: str, H: int, W: int):
        self.reference_dir = reference_dir
        self.decomp_type = decomp_type
        self.H = H
        self.W = W
        self.reference_data = {}
        
        if decomp_type == '3comp':
            self.file_mapping = {
                'ps': 'Freeman_Odd.bin',
                'pd': 'Freeman_Dbl.bin',
                'pv': 'Freeman_Vol.bin',
            }
        elif decomp_type == '4comp':
            self.file_mapping = {
                'ps': 'Yamaguchi4_Y4O_Odd.bin',
                'pd': 'Yamaguchi4_Y4O_Dbl.bin',
                'pv': 'Yamaguchi4_Y4O_Vol.bin',
                'ph': 'Yamaguchi4_Y4O_Hlx.bin',
            }
        elif decomp_type == '6comp':
            self.file_mapping = {
                'ps': 'Singh_i6SD_Odd.bin',
                'pd': 'Singh_i6SD_Dbl.bin',
                'pv': 'Singh_i6SD_Vol.bin',
                'ph': 'Singh_i6SD_Hlx.bin',
                'pod': 'Singh_i6SD_OD.bin',
                'pcd': 'Singh_i6SD_CD.bin',
            }
        else:
            raise ValueError(f"Unknown decomp_type: {decomp_type}")
        
        self._load_reference_data()
    
    def _load_reference_data(self):
        for name, filename in self.file_mapping.items():
            filepath = os.path.join(self.reference_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"⚠️ 参考文件不存在: {filepath}")
                self.reference_data[name] = None
                continue
            
            try:
                data = np.fromfile(filepath, dtype=np.float32)
                if len(data) != self.H * self.W:
                    print(f"❌ 数据大小不匹配: {filename} (预期{self.H*self.W}, 实际{len(data)})")
                    self.reference_data[name] = None
                    continue
                data = data.reshape(self.H, self.W)
                print(f"✅ 加载参考数据: {name} | 形状: {data.shape} | 范围: [{data.min():.4e}, {data.max():.4e}]")
                self.reference_data[name] = torch.from_numpy(data)
            except Exception as e:
                print(f"❌ 加载失败: {filename} | 错误: {e}")
                self.reference_data[name] = None
    
    def get_reference(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        if name in self.reference_data and self.reference_data[name] is not None:
            return self.reference_data[name].to(device)
        return None
    
    def get_reference_patch(self, name: str, device: torch.device,
                          h_start: int, w_start: int,
                          patch_h: int, patch_w: int) -> Optional[torch.Tensor]:
        ref_full = self.get_reference(name, device)
        if ref_full is None:
            return None
        return ref_full[h_start:h_start+patch_h, w_start:w_start+patch_w]


def _to_hermitian(T):
    if T.dim() == 3 and T.shape[1:] == (3, 3):
        T_mat = T if T.is_complex() else T.to(torch.complex64)
        return (T_mat + T_mat.conj().transpose(-2, -1)) / 2.0
    else:
        raise ValueError(f"Unsupported T shape: {T.shape}")



# ========================== Loss 1: 重构损失==========================
def reconstruction_loss(
    T_pred: torch.Tensor,
    T_target: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    T_pred = _to_hermitian(T_pred)
    T_target = _to_hermitian(T_target)
    
    device = T_pred.device
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
   
    diag_pred = torch.diagonal(T_pred.real, dim1=-2, dim2=-1)
    diag_target = torch.diagonal(T_target.real, dim1=-2, dim2=-1)
    
    diag_loss_linear = F.mse_loss(diag_pred, diag_target)
    diag_pred_log = torch.log10(torch.clamp(diag_pred, min=EPS))
    diag_target_log = torch.log10(torch.clamp(diag_target, min=EPS))
    diag_loss_log = F.mse_loss(diag_pred_log, diag_target_log)
    
    diag_loss = diag_loss_linear + 10.0 * diag_loss_log
    total_loss += diag_loss
    
    loss_dict['diag_linear'] = diag_loss_linear.item()
    loss_dict['diag_log'] = diag_loss_log.item()
    
    # 非对角线元素
    offdiag_loss = 0.0
    for i, j, name in [(0, 1, 'T12'), (0, 2, 'T13'), (1, 2, 'T23')]:
        real_loss = F.mse_loss(T_pred[:, i, j].real, T_target[:, i, j].real)
        imag_loss = F.mse_loss(T_pred[:, i, j].imag, T_target[:, i, j].imag)
        offdiag_loss += real_loss + imag_loss
        loss_dict[f'{name}_real'] = real_loss.item()
        loss_dict[f'{name}_imag'] = imag_loss.item()
    
    total_loss += 2.0 * offdiag_loss
    loss_dict['offdiag_total'] = (2.0 * offdiag_loss).item()
    loss_dict['total_recon'] = total_loss.item()
    
    return total_loss, loss_dict
    
    

# ========================== Loss 2: 参考值对比损失 ==========================
def robust_huber_loss(pred, ref, delta=0.1):
    """
    Huber Loss：对异常值更鲁棒
    
    原理：
    - 误差小时用平方（MSE）
    - 误差大时用绝对值（MAE）
    - 比MSE对离群点不敏感
    """
    diff = pred - ref
    abs_diff = torch.abs(diff)
    
    # Huber公式
    quadratic = torch.clamp(abs_diff, max=delta)
    linear = abs_diff - quadratic
    loss = 0.5 * quadratic**2 + delta * linear
    
    return torch.mean(loss)


def reference_comparison_loss(
    contrib: Dict[str, torch.Tensor],
    decomp_type: str,
    reference_loader: Optional[ReferenceDataLoader] = None,
    patch_coords: Optional[List[Tuple[int, int]]] = None,
    sample_ratio: float = 1,
    use_huber: bool = True, 
    huber_delta: float = 0.01, 
) -> Tuple[torch.Tensor, Dict[str, float]]:
  
    device = list(contrib.values())[0].device
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
    
    if reference_loader is None:
        loss_dict['total_reference'] = 0.0
        return total_loss, loss_dict
    
    if decomp_type == '3comp':
        power_names = ['ps', 'pd', 'pv']
    elif decomp_type == '4comp':
        power_names = ['ps', 'pd', 'pv', 'ph']
    elif decomp_type == '6comp':
        power_names = ['ps', 'pd', 'pv', 'ph', 'pod', 'pcd']
    else:
        power_names = []
    
    compared_count = 0

    for name in power_names:
        if name not in contrib:
            continue

        pred = contrib[name]
        B, patch_h, patch_w = pred.shape
    
        if patch_coords is not None:
            ref_patches = []
            for b in range(B):
                try:
                    if isinstance(patch_coords, (list, tuple)):
                        coord = patch_coords[b]
                        if isinstance(coord, (tuple, list)) and len(coord) == 2:
                            h_start, w_start = int(coord[0]), int(coord[1])
                        else:
                            h_start, w_start = int(coord[0]), int(coord[1])
                    else:
                        print(f"⚠️ 未知的坐标格式: {type(patch_coords)}")
                        break
                except Exception as e:
                    print(f"⚠️ 坐标解析失败: {e}")
                    break

                ref_patch = reference_loader.get_reference_patch(
                    name, device, h_start, w_start, patch_h, patch_w
                )
                if ref_patch is None:
                    break
                ref_patches.append(ref_patch)
            
            if len(ref_patches) != B:
                continue
            
            ref_data = torch.stack(ref_patches, dim=0)
            
        else:
            ref_full = reference_loader.get_reference(name, device)
            if ref_full is None:
                continue
            
            if ref_full.shape != (patch_h, patch_w):
                print(f"⚠️ 尺寸不匹配: pred={pred.shape}, ref={ref_full.shape}")
                continue
            
            ref_data = ref_full.unsqueeze(0).expand(B, -1, -1)
        
        N_pixels = patch_h * patch_w
        n_samples = int(N_pixels * sample_ratio)
        
        if sample_ratio >= 1.0:
            pred_flat = pred.reshape(B, -1)
            ref_flat = ref_data.reshape(B, -1)
        else:
            generator = torch.Generator(device=device).manual_seed(42)
            indices = torch.randperm(N_pixels, generator=generator, device=device)[:n_samples]
            
            pred_flat = pred.reshape(B, -1)[:, indices]
            ref_flat = ref_data.reshape(B, -1)[:, indices]
        
        pred_flat = torch.clamp(pred_flat, min=EPS)
        ref_flat = torch.clamp(ref_flat, min=EPS)
        
    
        if use_huber:
            mse_linear = robust_huber_loss(pred_flat, ref_flat, delta=huber_delta)
        else:
            mse_linear = F.mse_loss(pred_flat, ref_flat)
        
        pred_log = torch.log10(pred_flat)
        ref_log = torch.log10(ref_flat)
        
        if use_huber:
            mse_log = robust_huber_loss(pred_log, ref_log, delta=huber_delta)
        else:
            mse_log = F.mse_loss(pred_log, ref_log)
        
        component_loss = mse_linear + 10.0 * mse_log
        total_loss = total_loss + component_loss
        
        loss_dict[f'{name}_ref_linear'] = mse_linear.item()
        loss_dict[f'{name}_ref_log'] = mse_log.item()
        loss_dict[f'{name}_ref_total'] = component_loss.item()
        
        compared_count += 1
    
    if compared_count > 0:
        total_loss = total_loss / compared_count
    
    loss_dict['total_reference'] = total_loss.item()
    loss_dict['compared_components'] = compared_count
    
    return total_loss, loss_dict


# ========================== Loss 3: 平滑损失=========================
def smoothness_loss(
    contrib: Dict[str, torch.Tensor],
    decomp_type: str,
    T_target: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    device = list(contrib.values())[0].device
    total_tv = torch.tensor(0.0, device=device)
    loss_dict = {}

    if T_target is not None:

        sample_key = next(k for k in ['ps', 'pd', 'pv'] if k in contrib)
        B, H, W = contrib[sample_key].shape
        
        T11 = T_target[:, 0, 0].real  # [B*H*W]
        T22 = T_target[:, 1, 1].real  # [B*H*W]
        T33 = T_target[:, 2, 2].real

        span = (T11 + T22 + T33).reshape(B, H, W)


        edge_h_2d = torch.exp(-5.0 * torch.abs(span[:, :, 1:] - span[:, :, :-1]))
        edge_v_2d = torch.exp(-5.0 * torch.abs(span[:, 1:, :] - span[:, :-1, :]))
    
        edge_h = edge_h_2d[:, :-1, :].unsqueeze(1)  
        edge_v = edge_v_2d[:, :, :-1].unsqueeze(1)
    
    else:
        edge_h = edge_v = 1.0

    power_keys = ['ps', 'pd', 'pv']
    if decomp_type in ['4comp', '6comp']:
        power_keys.append('ph')
    if decomp_type == '6comp':
        power_keys.extend(['pod', 'pcd'])

    count = 0
    for k in power_keys:
        if k not in contrib:
            continue
        x = contrib[k]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        dh = x[:, :, :, 1:] - x[:, :, :, :-1]
        dv = x[:, :, 1:, :] - x[:, :, :-1, :]

        tv = torch.mean(
            edge_h * torch.abs(dh[:, :, :-1, :]) +
            edge_v * torch.abs(dv[:, :, :, :-1]) +
            epsilon
        )

        total_tv += tv
        loss_dict[f'tv_{k}'] = tv.item()
        count += 1

    if count > 0:
        total_tv /= count

    loss_dict['tv_total'] = total_tv.item()
    return total_tv, loss_dict


# ========================== 组合损失函数 ==========================
def combined_loss(
    T_pred: torch.Tensor,
    T_target: torch.Tensor,
    contrib: Dict[str, torch.Tensor],
    decomp_type: str,
    recon_lambda: float = 1.0,
    reference_lambda: float = 0.5,
    smooth_lambda: float = 0.05,
    reference_loader: Optional[ReferenceDataLoader] = None,
    patch_coords: Optional[List[Tuple[int, int]]] = None,
    sample_ratio: float = 0.2,
    use_huber: bool = True,  
    huber_delta: float = 0.1,  
    **kwargs
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    组合损失函数：3个loss的加权和
    
    改进：
    - 参考损失使用Huber Loss，对异常值更鲁棒
    - 其他保持不变
    """
    loss_dict = {}
    
    # Loss 1: 重构损失
    recon_loss, recon_dict = reconstruction_loss(T_pred, T_target)
    loss_dict.update({f'recon_{k}': v for k, v in recon_dict.items()})
    
    # Loss 2: 参考值对比损失
    ref_loss, ref_dict = reference_comparison_loss(
        contrib, decomp_type, reference_loader, patch_coords, 
        sample_ratio, use_huber, huber_delta
    )
    loss_dict.update({f'ref_{k}': v for k, v in ref_dict.items()})
    
    # Loss 3: 平滑损失
    smooth_loss, smooth_dict = smoothness_loss(contrib, decomp_type, T_target)
    loss_dict.update({f'smooth_{k}': v for k, v in smooth_dict.items()})
    
    # 总损失
    total_loss = (
        recon_lambda * recon_loss + 
        reference_lambda * ref_loss + 
        smooth_lambda * smooth_loss
    )
    
    loss_dict['weighted_recon'] = (recon_lambda * recon_loss).item()
    loss_dict['weighted_reference'] = (reference_lambda * ref_loss).item()
    loss_dict['weighted_smooth'] = (smooth_lambda * smooth_loss).item()
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


# ========================== 工厂函数 ==========================
def get_loss_fn(model_type: ModelType, reference_dir: str = None, H: int = None, W: int = None) -> callable:
  
    decomp_str = model_type.value
    
    ref_loader = None
    if reference_dir is not None and H is not None and W is not None:
        try:
            ref_loader = ReferenceDataLoader(reference_dir, decomp_str, H, W)
            print(f"✅ 参考数据加载器初始化成功 | 类型: {decomp_str}")
        except Exception as e:
            print(f"⚠️ 参考数据加载失败: {e}")
            ref_loader = None

    def loss_fn(T_pred, T_target, contrib, patch_coords=None, **kwargs):
        return combined_loss(
            T_pred=T_pred,
            T_target=T_target,
            contrib=contrib,
            decomp_type=decomp_str,
            reference_loader=ref_loader,
            patch_coords=patch_coords,
            **kwargs
        )

    return loss_fn