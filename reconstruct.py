#!/usr/bin/env python3

import torch
from typing import Optional, Tuple, Dict
import component  

def rotate_to_deoriented_space(T: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    将观测空间的T矩阵旋转到去取向空间
    
    T: [B, 3, 3, H, W] 观测空间
    theta: [B, H, W] 旋转角度
    返回: [B, 3, 3, H, W] 去取向空间
    
    数学: T_derot = R(θ)^H @ T @ R(θ)
    """
    B, _, _, H, W = T.shape
    device = T.device
    
    cos_2theta = torch.cos(2 * theta)
    sin_2theta = torch.sin(2 * theta)
    
    # 构造旋转矩阵 R(θ)
    R = torch.zeros((B, H, W, 3, 3), dtype=torch.complex64, device=device)
    R[:, :, :, 0, 0] = 1.0
    R[:, :, :, 1, 1] = cos_2theta
    R[:, :, :, 1, 2] = sin_2theta
    R[:, :, :, 2, 1] = -sin_2theta
    R[:, :, :, 2, 2] = cos_2theta
    
    R_h = R.conj().transpose(-2, -1)
    
    # T转换维度
    T_perm = T.permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 3]
    
    # 去旋转
    T_derot = torch.matmul(torch.matmul(R_h, T_perm), R)
    
    # 恢复维度
    return T_derot.permute(0, 3, 4, 1, 2)

def reconstruct_3comp(ps, pd, pv, beta, alpha, weights=None, **kwargs):


    ps = ps[0]
    pd = pd[0]
    pv = pv[0]
    beta = beta[0]
    alpha = alpha[0]

    Ts = component.surface_component(ps, beta)
    Td = component.double_bounce_component(pd, alpha)
    Tv = component.volume_component1(pv)  

    T_recon = Ts + Td + Tv


    # contrib
    contrib = {
        'ps': ps, 'pd': pd, 'pv': pv,
        'beta': beta, 'alpha': alpha, 
    }

    return T_recon, contrib


def reconstruct_4comp(ps, pd, pv, ph, beta, alpha, weights=None, T_target=None, **kwargs):

    E, N = ps.shape
    device = ps.device

    if weights is None:
        weights = torch.ones(E, N, device=device) / E

    if T_target is not None:

        T23_imag = T_target[:, 1, 2, :, :].imag  # [B, H, W]
        sign_h_theory = torch.sign(T23_imag).flatten()  # [N] ∈ {-1, 0, +1}
    else:
        
        sign_h_theory = torch.ones(N, device=device)
    
    T_recon = torch.zeros(N, 3, 3, dtype=torch.complex64, device=device)
    weights_exp = weights.unsqueeze(-1).unsqueeze(-1)

    for e in range(E):
       
        Ts_e = component.surface_component(ps[e], beta[e])
        Td_e = component.double_bounce_component(pd[e], alpha[e])
        Th_e = component.helix_component(ph[e], sign_h_theory)
        
        # 体散射（不同expert用不同模型）
        if e == 0:
            Tv_e = component.volume_component1(pv[e])
        elif e == 1:
            Tv_e = component.volume_component2(pv[e])
        else:  
            Tv_e = component.volume_component3(pv[e])
        
        T_e = Ts_e + Td_e + Th_e + Tv_e
        T_recon += T_e * weights_exp[e]
        
        # 及时释放显存
        del Ts_e, Td_e, Th_e, Tv_e, T_e
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 加权平均参数
    contrib = {
        'ps': torch.sum(ps * weights, dim=0),
        'pd': torch.sum(pd * weights, dim=0),
        'pv': torch.sum(pv * weights, dim=0),
        'ph': torch.sum(ph * weights, dim=0),
        'beta': torch.sum(beta * weights, dim=0),
        'alpha': torch.sum(alpha * weights, dim=0),
        'weights': weights,
        'sign_h': sign_h_theory,
    }
    return T_recon, contrib
    

def reconstruct_6comp(ps, pd, pv, ph, pod, pcd, beta, alpha,
                      weights=None, theta=None, T_target=None, **kwargs):
    
    E, N = ps.shape
    device = ps.device

    if weights is None:
        weights = torch.ones(E, N, device=device) / E
    
    B = kwargs.get('B', 1)
    H = kwargs.get('H', 1)
    W = kwargs.get('W', 1)
    
    if T_target is not None and theta is not None and theta.numel() == N:
        # Reshape theta
        theta_image = theta.reshape(B, H, W)
        
        # 将观测空间的T_target旋转到去取向空间
        T_target_derotated = rotate_to_deoriented_space(T_target, theta_image)
        
        # ==================== 步骤2: 从去取向空间提取符号 ====================
        # sign_h（来自去取向空间的T23虚部）
        T23_imag_derot = T_target_derotated[:, 1, 2, :, :].imag
        sign_h_theory = torch.sign(T23_imag_derot).flatten()
        
        # sign_od/sign_cd（来自去取向空间的T13）
        T13_real_derot = T_target_derotated[:, 0, 2, :, :].real
        T13_imag_derot = T_target_derotated[:, 0, 2, :, :].imag
        
        sign_od = torch.sign(T13_real_derot).flatten()
        sign_cd = torch.sign(T13_imag_derot).flatten()
        sign_odcd_theory = (sign_od + sign_cd) / 2.0
    else:
        # 默认符号
        sign_h_theory = torch.ones(N, device=device)
        sign_odcd_theory = torch.ones(N, device=device)
        T_target_derotated = T_target if T_target is not None else None
    
    # ==================== 步骤3: 在去取向空间重构 ====================
    T_recon_derot = torch.zeros(N, 3, 3, dtype=torch.complex64, device=device)
    weights_exp = weights.unsqueeze(-1).unsqueeze(-1)
    
    for e in range(E):
        # 6个分量（使用从去取向空间提取的符号）
        Ts_e = component.surface_component(ps[e], beta[e])
        Td_e = component.double_bounce_component(pd[e], alpha[e])
        Th_e = component.helix_component(ph[e], sign_h_theory)
        Tod_e = component.od_component(pod[e], sign_odcd_theory)
        Tcd_e = component.cd_component(pcd[e], sign_odcd_theory)
        
        # 体散射
        if e == 0:
            Tv_e = component.volume_component1(pv[e])
        elif e == 1:
            Tv_e = component.volume_component2(pv[e])
        elif e == 2:
            Tv_e = component.volume_component3(pv[e])
        else:
            Tv_e = component.volume_component4(pv[e])
        
        # 在去取向空间求和
        T_e_derot = Ts_e + Td_e + Th_e + Tv_e + Tod_e + Tcd_e
        T_recon_derot += T_e_derot * weights_exp[e]
        
        del Ts_e, Td_e, Th_e, Tod_e, Tcd_e, Tv_e, T_e_derot
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    contrib = {
        'ps': torch.sum(ps * weights, dim=0),
        'pd': torch.sum(pd * weights, dim=0),
        'pv': torch.sum(pv * weights, dim=0),
        'ph': torch.sum(ph * weights, dim=0),
        'pod': torch.sum(pod * weights, dim=0),
        'pcd': torch.sum(pcd * weights, dim=0),
        'beta': torch.sum(beta * weights, dim=0),
        'alpha': torch.sum(alpha * weights, dim=0),
        'weights': weights,
        'sign_h': sign_h_theory,
        'sign_odcd': sign_odcd_theory,
        'theta': theta.reshape(B, H, W) if theta is not None else None,
        'T_target_derotated': T_target_derotated,  
    }
    
    # ==================== 步骤4: 返回去取向空间的结果 ====================
    return T_recon_derot, contrib
