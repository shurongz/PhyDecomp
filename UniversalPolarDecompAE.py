#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import reconstruct as rs

# ========== 改进的残差块：加入通道注意力 ==========
class LightSEResBlock(nn.Module):
    """优化：只保留通道注意力，去掉空间门控"""
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect')
        self.gn1 = nn.GroupNorm(8, dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect')
        self.gn2 = nn.GroupNorm(8, dim)

        # ✅ 只保留通道注意力（SE）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        out = F.leaky_relu(self.gn1(self.conv1(x)), 0.1)
        out = self.gn2(self.conv2(out))
        out = out * self.se(out)  # ✅ 去掉空间门控
        return F.leaky_relu(out + identity, 0.1)



# ========== 主模型 ==========
class CNNPolDecompAE(nn.Module):    

    def __init__(self, decomp_type: str = '4comp'):
        super().__init__()
        self.decomp_type = decomp_type.lower()
        self.base_dim = 64
        d = self.base_dim
        
        config = {
            '3comp': {'num_experts': 1},
            '4comp': {'num_experts': 3},
            '6comp': {'num_experts': 4},
        }
        self.num_experts = config.get(self.decomp_type, config['3comp'])['num_experts']
        
        # ========== Encoder（用SE残差块替换普通残差块）==========
        self.down1 = nn.Sequential(
            nn.Conv2d(9, d, 3, padding=1),
            LightSEResBlock(d),
            nn.Conv2d(d, d, 3, stride=2, padding=1)
        )
        self.down2 = nn.Sequential(
            LightSEResBlock(d),
            nn.Conv2d(d, d*2, 3, stride=2, padding=1)
        )
        self.down3 = nn.Sequential(
            LightSEResBlock(d*2),
            nn.Conv2d(d*2, d*4, 3, stride=2, padding=1)
        )
        
        # ========== Bottleneck + 改进的MoE ==========
        self.bottleneck = LightSEResBlock(d*4)
        
        # 门控网络（加入更多非线性）
        self.gate_net = nn.Sequential(
            nn.Conv2d(d*4, d*2, 1),
            nn.GroupNorm(4, d*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d*2, self.num_experts, 1)
        )
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Experts
        self.experts = nn.ModuleList([
            LightSEResBlock(d*4)  # ✅ 从2个减少到1个
            for _ in range(self.num_experts)
        ])
        
        # ========== Decoder（用SE残差块）==========
        self.feature_refine = nn.Conv2d(d*4, d, 1)
        
        # ========== 参数解耦分支（粗/细尺度分离）==========
       
        self.param_branch = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1),
            nn.GroupNorm(8, d),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
  
        self.power_heads = nn.ModuleDict({
            'ps': nn.Conv2d(d, 1, 1),
            'pd': nn.Conv2d(d, 1, 1),
            'pv': nn.Conv2d(d, 1, 1),
        })
        
    
        if self.decomp_type in ['4comp', '6comp']:
            self.power_heads['ph'] = nn.Conv2d(d, 1, 1)
        if self.decomp_type == '6comp':
            self.power_heads['pod'] = nn.Conv2d(d, 1, 1)
            self.power_heads['pcd'] = nn.Conv2d(d, 1, 1)
        
     
        self.beta_head = nn.Conv2d(d, 2, 1)
        self.alpha_head = nn.Conv2d(d, 2, 1)
        if self.decomp_type == '6comp':
            self.theta_head = nn.Conv2d(d, 1, 1)
        
        self._init_weights()
        
    
    def _init_weights(self):
        init_bias = {'ps': -0.8, 'pd': -1.0, 'pv': 0.2, 'ph': -2.5, 'pod': -2.0, 'pcd': -2.0}
        for name, head in self.power_heads.items():
            nn.init.constant_(head.bias, init_bias.get(name, -1.0))
            nn.init.normal_(head.weight, 0.0, 0.02)
         
    
    def _reconstruct(self, params, weights_up, B, H, W, T_target=None):
        N = B * H * W
        rk = {'B': B, 'H': H, 'W': W}
        
        def flatten_v(t):
            if t is None: return None
            return t.reshape(-1).unsqueeze(0).repeat(self.num_experts, 1)
        
        f_ps = flatten_v(params['ps'])
        f_pd = flatten_v(params['pd'])
        f_pv = flatten_v(params['pv'])
        f_ph = flatten_v(params['ph'])
        f_pod = flatten_v(params['pod'])
        f_pcd = flatten_v(params['pcd'])
        f_beta = flatten_v(params['beta'])
        f_alpha = flatten_v(params['alpha'])
        f_theta = params['theta'].flatten() if params['theta'] is not None else None
        f_weights = weights_up.permute(1, 0, 2, 3).reshape(self.num_experts, -1)
        
        if self.decomp_type == '3comp':
            T_recon_f, contrib_f = rs.reconstruct_3comp(
                ps=f_ps, pd=f_pd, pv=f_pv, beta=f_beta, alpha=f_alpha,
                weights=f_weights, **rk
            )
        elif self.decomp_type == '4comp':
            T_recon_f, contrib_f = rs.reconstruct_4comp(
                ps=f_ps, pd=f_pd, pv=f_pv, ph=f_ph, beta=f_beta, alpha=f_alpha,
                weights=f_weights, T_target=T_target, **rk
            )
        elif self.decomp_type == '6comp':
            T_recon_f, contrib_f = rs.reconstruct_6comp(
                ps=f_ps, pd=f_pd, pv=f_pv, ph=f_ph, pod=f_pod, pcd=f_pcd,
                beta=f_beta, alpha=f_alpha, theta=f_theta,
                weights=f_weights, T_target=T_target, **rk
            )
        
        T_recon = T_recon_f.reshape(B, H, W, 3, 3).permute(0, 3, 4, 1, 2)
        return T_recon, contrib_f
    
    def forward(self, x: torch.Tensor, T_target: torch.Tensor = None, is_training: bool = True):
        B, _, H_orig, W_orig = x.shape
        N_pixels = B * H_orig * W_orig
        
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        bottle = self.bottleneck(d3)


        gate_logits = self.gate_net(bottle)

        temp = F.softplus(self.temperature) + 0.5
        
        if is_training and self.num_experts > 1:
            weights = F.gumbel_softmax(gate_logits, tau=temp, hard=True, dim=1)
        elif self.num_experts > 1:
            idx = torch.argmax(gate_logits, dim=1, keepdim=True)
            weights = torch.zeros_like(gate_logits).scatter_(1, idx, 1.0)
        else:
            weights = torch.ones_like(gate_logits)

        if self.num_experts == 1:
            moe_feat = self.experts[0](bottle)
        else:
            moe_feat = sum(
                self.experts[i](bottle) * weights[:, i:i+1]
                for i in range(self.num_experts)
            )
        
        feat_refined = self.feature_refine(moe_feat)

        feat_refined = F.interpolate(
            feat_refined, 
            size=(H_orig, W_orig), 
            mode='bilinear', 
            align_corners=False
        )
        
        
        # 参数解耦：功率 vs 相位
        param_feat = self.param_branch(feat_refined)
        
    
        main_params = {
            'ps': F.softplus(self.power_heads['ps'](param_feat)).squeeze(1) + 1e-6,
            'pd': F.softplus(self.power_heads['pd'](param_feat)).squeeze(1) + 1e-6,
            'pv': F.softplus(self.power_heads['pv'](param_feat)).squeeze(1) + 1e-6,
            'ph': None,
            'pod': None,
            'pcd': None,
        }
        
        if 'ph' in self.power_heads:
            main_params['ph'] = F.softplus(self.power_heads['ph'](param_feat)).squeeze(1) + 1e-6
        if 'pod' in self.power_heads:
            main_params['pod'] = F.softplus(self.power_heads['pod'](param_feat)).squeeze(1) + 1e-6
        if 'pcd' in self.power_heads:
            main_params['pcd'] = F.softplus(self.power_heads['pcd'](param_feat)).squeeze(1) + 1e-6
        
        beta_raw = self.beta_head(param_feat)
        beta_c = torch.complex(beta_raw[:, 0], beta_raw[:, 1])
        main_params['beta'] = torch.polar(torch.tanh(torch.abs(beta_c)), torch.angle(beta_c))
        
        alpha_raw = self.alpha_head(param_feat)
        alpha_c = torch.complex(alpha_raw[:, 0], alpha_raw[:, 1])
        main_params['alpha'] = torch.polar(torch.tanh(torch.abs(alpha_c)), torch.angle(alpha_c))
        
        main_params['theta'] = None
        if self.decomp_type == '6comp':
            main_params['theta'] = (torch.pi / 2) * torch.tanh(self.theta_head(param_feat).squeeze(1))
        
        # 重构
        w_up = F.interpolate(weights, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        T_recon, contrib_f = self._reconstruct(main_params, w_up, B, H_orig, W_orig, T_target)
        

        contrib = {}
        for k, v in contrib_f.items():
            if v is not None and v.numel() == N_pixels:
                contrib[k] = v.view(B, H_orig, W_orig)
        
        contrib['beta_real'] = main_params['beta'].real
        contrib['beta_imag'] = main_params['beta'].imag
        contrib['alpha_real'] = main_params['alpha'].real
        contrib['alpha_imag'] = main_params['alpha'].imag
        if main_params['theta'] is not None:
            contrib['theta'] = main_params['theta']
        contrib['gate_logits'] = gate_logits
        
        return T_recon, contrib, weights


 