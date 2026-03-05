#!/usr/bin/env python3
import torch

def surface_component(ps: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    表面散射（Bragg散射）
    T_s = (ps / (1+|β|²)) * [[1, β*], [β, |β|²]]
    """
    N = ps.shape[0]
    device = ps.device
    Ts = torch.zeros((N, 3, 3), dtype=torch.complex64, device=device)
    
    beta_mag_sq = torch.abs(beta) ** 2
    scale = ps / (1.0 + beta_mag_sq)
    
    Ts[:, 0, 0] = scale
    Ts[:, 1, 1] = scale * beta_mag_sq
    Ts[:, 0, 1] = scale * torch.conj(beta)
    Ts[:, 1, 0] = torch.conj(Ts[:, 0, 1])
    return Ts

def double_bounce_component(pd: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    二面角散射
    T_d = (pd / (1+|α|²)) * [[|α|², α], [α*, 1]]
    """
    N = pd.shape[0]
    device = pd.device
    Td = torch.zeros((N, 3, 3), dtype=torch.complex64, device=device)
    
    alpha_mag_sq = torch.abs(alpha) ** 2
    scale = pd / (1.0 + alpha_mag_sq)
    
    Td[:, 0, 0] = scale * alpha_mag_sq
    Td[:, 1, 1] = scale
    Td[:, 0, 1] = scale * alpha
    Td[:, 1, 0] = torch.conj(Td[:, 0, 1])
    return Td

def volume_component1(pv: torch.Tensor) -> torch.Tensor:
    """随机体散射模型1（标准模型）"""
    N = pv.shape[0]
    Tv = torch.zeros((N, 3, 3), dtype=torch.complex64, device=pv.device)
    scale = pv / 4.0
    
    Tv[:, 0, 0] = 2 * scale
    Tv[:, 1, 1] = scale
    Tv[:, 2, 2] = scale
    return Tv

def volume_component2(pv: torch.Tensor) -> torch.Tensor:
    """随机体散射模型2"""
    N = pv.shape[0]
    Tv = torch.zeros((N, 3, 3), dtype=torch.complex64, device=pv.device)
    scale = pv / 30.0
    
    Tv[:, 0, 0] = 15 * scale
    Tv[:, 1, 1] = 7 * scale
    Tv[:, 2, 2] = 8 * scale
    Tv[:, 0, 1] = 5 * scale
    Tv[:, 1, 0] = Tv[:, 0, 1]  
    return Tv

def volume_component3(pv: torch.Tensor) -> torch.Tensor:
    """随机体散射模型3"""
    N = pv.shape[0]
    Tv = torch.zeros((N, 3, 3), dtype=torch.complex64, device=pv.device)
    scale = pv / 30.0
    
    Tv[:, 0, 0] = 15 * scale
    Tv[:, 1, 1] = 7 * scale
    Tv[:, 2, 2] = 8 * scale
    Tv[:, 0, 1] = -5 * scale
    Tv[:, 1, 0] = Tv[:, 0, 1]
    return Tv

def volume_component4(pv: torch.Tensor) -> torch.Tensor:
    """随机体散射模型4"""
    N = pv.shape[0]
    Tv = torch.zeros((N, 3, 3), dtype=torch.complex64, device=pv.device)
    scale = pv / 15.0
    
    Tv[:, 1, 1] = 7 * scale
    Tv[:, 2, 2] = 8 * scale
    return Tv

def helix_component(ph: torch.Tensor, sign_h: torch.Tensor = None) -> torch.Tensor:
    """螺旋散射（sign_h从T矩阵提取）"""
    N = ph.shape[0]
    device = ph.device
    Th = torch.zeros((N, 3, 3), dtype=torch.complex64, device=device)
    
    scale = ph / 2.0
    
    if sign_h is None:
        sign_h = torch.ones(N, device=device)
    elif not isinstance(sign_h, torch.Tensor):
        sign_h = torch.full((N,), float(sign_h), device=device)
    
    Th[:, 1, 1] = scale
    Th[:, 2, 2] = scale
    Th[:, 1, 2] = scale * sign_h * 1j
    Th[:, 2, 1] = -Th[:, 1, 2]
    return Th


def od_component(pod: torch.Tensor, sign_odcd: torch.Tensor = None) -> torch.Tensor:
    """Oriented Dipole"""
    N = pod.shape[0]
    device = pod.device
    Tod = torch.zeros((N, 3, 3), dtype=torch.complex64, device=device)
    
    scale = pod / 2.0
    
    if sign_odcd is None:
        sign_odcd = torch.ones(N, device=device)
    elif not isinstance(sign_odcd, torch.Tensor):
        sign_odcd = torch.full((N,), float(sign_odcd), device=device)
    
    Tod[:, 0, 0] = scale
    Tod[:, 2, 2] = scale
    Tod[:, 0, 2] = scale * sign_odcd
    Tod[:, 2, 0] = Tod[:, 0, 2]
    return Tod

def cd_component(pcd: torch.Tensor, sign_odcd: torch.Tensor = None) -> torch.Tensor:
    """Compound Dipole"""
    N = pcd.shape[0]
    device = pcd.device
    Tcd = torch.zeros((N, 3, 3), dtype=torch.complex64, device=device)
    
    scale = pcd / 2.0
    
    if sign_odcd is None:
        sign_odcd = torch.ones(N, device=device)
    elif not isinstance(sign_odcd, torch.Tensor):
        sign_odcd = torch.full((N,), float(sign_odcd), device=device)
    
    Tcd[:, 0, 0] = scale
    Tcd[:, 2, 2] = scale
    Tcd[:, 0, 2] = scale * sign_odcd * 1j
    Tcd[:, 2, 0] = -Tcd[:, 0, 2]
    return Tcd

def derotate_coherency(T_de: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:

    B, _, _, H, W = T_de.shape
    device = T_de.device
    
    cos_theta = torch.cos(-theta)
    sin_theta = torch.sin(-theta)
    
    # 构造旋转矩阵 R [B, H, W, 3, 3]
    R = torch.zeros((B, H, W, 3, 3), dtype=torch.complex64, device=device)
    R[:, :, :, 0, 0] = 1.0
    R[:, :, :, 1, 1] = cos_theta
    R[:, :, :, 1, 2] = sin_theta
    R[:, :, :, 2, 1] = -sin_theta
    R[:, :, :, 2, 2] = cos_theta
    
    R_h = R.conj().transpose(-2, -1)
    
    # T_de转换维度 [B, 3, 3, H, W] -> [B, H, W, 3, 3]
    T_de_perm = T_de.permute(0, 3, 4, 1, 2)
    
    # 矩阵乘法 T = R @ T_de @ R^H
    T_orig = torch.matmul(torch.matmul(R, T_de_perm), R_h)
    
    # 恢复维度 [B, H, W, 3, 3] -> [B, 3, 3, H, W]
    return T_orig.permute(0, 3, 4, 1, 2)

def derotate_coherency_batch(T: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
   
    return derotate_coherency(T, theta)