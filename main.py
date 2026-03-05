import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import argparse

from config import Config
from UniversalPolarDecompAE import CNNPolDecompAE
from dataset import PolSARDataset, polsar_collate_fn
from loss import ModelType, get_loss_fn


def unflatten_t3_image(feat: torch.Tensor) -> torch.Tensor:
    """
    辅助函数：将 Dataset 返回的 9通道特征 还原为 复数相干矩阵 T3
    输入: [B, 9, H, W]
    输出: [B, 3, 3, H, W] (Complex64)
    """
    B, C, H, W = feat.shape
    T = torch.zeros(B, 3, 3, H, W, dtype=torch.complex64, device=feat.device)
    
    # 对角线 (实数)
    T[:, 0, 0] = feat[:, 0]
    T[:, 1, 1] = feat[:, 1]
    T[:, 2, 2] = feat[:, 2]
    
    # 非对角线 (复数)
    T[:, 0, 1] = torch.complex(feat[:, 3], feat[:, 4])
    T[:, 1, 0] = torch.conj(T[:, 0, 1])
    T[:, 0, 2] = torch.complex(feat[:, 5], feat[:, 6])
    T[:, 2, 0] = torch.conj(T[:, 0, 2])
    T[:, 1, 2] = torch.complex(feat[:, 7], feat[:, 8])
    T[:, 2, 1] = torch.conj(T[:, 1, 2])
    
    return T

def save_bin(tensor: torch.Tensor, name: str, output_dir: str, model_type: str, scale_factor: float = 1.0):
    """保存bin文件（命名规则：[模型类型]_[分量类型]_[名称].bin）"""
    os.makedirs(output_dir, exist_ok=True)
    data = tensor.detach().cpu().numpy().astype(np.float32).squeeze()

    if scale_factor != 1.0:
        data = data / scale_factor
        print(f"   📏 还原量级: scale_factor = {scale_factor:.4e}")
    
    filename = f"{model_type}_{name}.bin"
    path = os.path.join(output_dir, filename)
    
    data.tofile(path)
    print(f"   ✅ 保存 {filename} | 形状: {data.shape}")

def print_batch_statistics(contrib: dict, model_type: str, remove_outliers: bool = True):
    """
    
    参数:
        contrib: 模型输出的参数字典
        model_type: 模型类型 ('3comp'/'4comp'/'6comp')
        remove_outliers: 是否剔除极端值（使用1%和99%分位数）
    """
    print("\n" + "="*80)
    print(f"📊 Epoch统计 | 模型: {model_type} | 剔除极端值: {remove_outliers}")
    print("="*80)
    
    # 根据模型类型确定要输出的功率分量
    if model_type == '3comp':
        power_keys = ['ps', 'pd', 'pv']
        param_keys = ['alpha_real', 'alpha_imag', 'beta_real', 'beta_imag']
    elif model_type == '4comp':
        power_keys = ['ps', 'pd', 'pv', 'ph']
        param_keys = ['alpha_real', 'alpha_imag', 'beta_real', 'beta_imag']
    elif model_type == '6comp':
        power_keys = ['ps', 'pd', 'pv', 'ph', 'pod', 'pcd']
        param_keys = ['alpha_real', 'alpha_imag', 'beta_real', 'beta_imag']
    else:
        power_keys = []
        param_keys = []
    
    # ==================== 1. 功率分量统计 ====================
    print("\n【功率分量】")
    print(f"{'名称':<8} {'均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'中位数':>10}")
    print("-" * 80)
    
    for key in power_keys:
        if key in contrib:
            val = contrib[key].detach().cpu().flatten()
            
            if remove_outliers:
                # 剔除1%和99%分位数外的值
                low = torch.quantile(val, 0.01)
                high = torch.quantile(val, 0.99)
                val_filtered = val[(val >= low) & (val <= high)]
            else:
                val_filtered = val
            
            mean = val_filtered.mean().item()
            std = val_filtered.std().item()
            vmin = val_filtered.min().item()
            vmax = val_filtered.max().item()
            median = val_filtered.median().item()
            
            print(f"{key:<8} {mean:>10.4f} {std:>10.4f} {vmin:>10.4f} {vmax:>10.4f} {median:>10.4f}")
    
    # ==================== 2. 功率占比统计 ====================
    print("\n【功率占比（相对于总功率）】")
    print(f"{'名称':<8} {'占比均值':>12} {'占比标准差':>12} {'占比范围':>20}")
    print("-" * 80)
    
    power_components = []
    for key in power_keys:
        if key in contrib:
            power_components.append(contrib[key].reshape(-1))
    
    if len(power_components) > 0:
        all_powers = torch.stack(power_components, dim=1)  # [N, K]
        total_power = torch.sum(all_powers, dim=1, keepdim=True) + 1e-8
        ratios = all_powers / total_power  # [N, K]
        
        for i, key in enumerate(power_keys):
            if key in contrib:
                ratio_mean = ratios[:, i].mean().item()
                ratio_std = ratios[:, i].std().item()
                ratio_min = ratios[:, i].min().item()
                ratio_max = ratios[:, i].max().item()
                print(f"{key:<8} {ratio_mean*100:>11.2f}% {ratio_std*100:>11.2f}% "
                      f"[{ratio_min*100:>6.2f}%, {ratio_max*100:>6.2f}%]")
    
    print("="*80)

def print_final_T_values(pred_mat: torch.Tensor, target_mat: torch.Tensor):
    """只打印最终的预测/输入T矩阵元素值（均值+范围）"""
    print("\n" + "="*100)
    print(f"📊 最终T矩阵元素值统计 | 总像素数: {pred_mat.shape[0]:,}")
    print("="*100)
    
    # 对角线元素（实数）
    print("\n【对角线元素（实数）】")
    diag_info = [
        ("T11", 0, 0), ("T22", 1, 1), ("T33", 2, 2)
    ]
    for name, i, j in diag_info:
        pred_val = pred_mat[:, i, j].real
        target_val = target_mat[:, i, j].real
        print(f"{name}:")
        print(f"  预测值 - 均值: {pred_val.mean():.6f} | 范围: [{pred_val.min():.6f}, {pred_val.max():.6f}]")
        print(f"  输入值 - 均值: {target_val.mean():.6f} | 范围: [{target_val.min():.6f}, {target_val.max():.6f}]")
        
    print("\n【非对角线元素（复数）】")
    non_diag_info = [("T12", 0, 1), ("T13", 0, 2), ("T23", 1, 2)]
    for name, i, j in non_diag_info:
        pred_real = pred_mat[:, i, j].real
        pred_imag = pred_mat[:, i, j].imag
        target_real = target_mat[:, i, j].real
        target_imag = target_mat[:, i, j].imag
        
        print(f"{name}:")
        print(f"  预测值 - 实部均值: {pred_real.mean():.6f} | 虚部均值: {pred_imag.mean():.6f}")
        print(f"  输入值 - 实部均值: {target_real.mean():.6f} | 虚部均值: {target_imag.mean():.6f}")
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description='PolSAR Decomposition Training & Inference')
    parser.add_argument('--model_type', type=str, default='3comp', choices=Config.DECOMP_TYPES)
    parser.add_argument('--data_dir', type=str, default=None,help=f'数据目录（默认: {Config.DATA_DIR}）')
    parser.add_argument('--output_dir', type=str, default=None,help=f'输出目录（默认: {Config.OUTPUT_DIR}）')
    parser.add_argument('--batch_size', type=int, default=Config.DEFAULT_BATCH_SIZE)
    parser.add_argument('--patch_size', type=int, default=Config.DEFAULT_PATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=Config.DEFAULT_EPOCHS)
    parser.add_argument('--lr', type=float, default=Config.DEFAULT_LR)
    parser.add_argument('--inference', action='store_true', help='仅推理模式')
    parser.add_argument('--full_image', action='store_true', default=True,
                        help='使用全图')
    parser.add_argument('--roi', type=int, nargs=4, default=None,
                        metavar=('ROW_START', 'ROW_END', 'COL_START', 'COL_END'),
                        help='ROI区域')
    parser.add_argument('--recon_lambda', type=float, default=Config.DEFAULT_RECON_LAMBDA,
                        help='重构损失权重')
    parser.add_argument('--reference_lambda', type=float, default=Config.DEFAULT_REFERENCE_LAMBDA,
                        help='物理约束权重')
    parser.add_argument('--smooth_lambda', type=float, default=Config.DEFAULT_SMOOTH_LAMBDA,
                        help='平滑损失权重（TV正则）')
    parser.add_argument('--sample_ratio', type=float, default=Config.DEFAULT_SAMPLE_RATIO,
                        help='参考值采样比例')
    parser.add_argument('--reference_dir', type=str, default=Config.REFERENCE_DIR,
                        help='参考数据目录')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='每N个batch打印一次统计（0=不打印）')
    parser.add_argument('--remove_outliers', action='store_true', default=True,
                        help='统计时剔除极端值')
    args = parser.parse_args()
    
    
    data_dir = args.data_dir if args.data_dir else Config.DATA_DIR
    output_dir = args.output_dir if args.output_dir else Config.OUTPUT_DIR
    
    # 创建输出目录
    Config.create_output_dirs()
    
    # 验证路径
    Config.validate_paths()
    
    # ROI处理
    roi = None if args.full_image else args.roi
    
    # 模型类型
    model_type_str = args.model_type
    model_type_enum = ModelType[f'comp{model_type_str[0]}']
    num_experts = Config.NUM_EXPERTS[model_type_str]

    print("="*70)
    print("PolSAR分解训练脚本 | 统一配置管理")
    print("="*70)
    print(f"模型类型: {model_type_str} | 专家数量: {num_experts}")
    print(f"数据目录: {data_dir}")
    print(f"参考目录: {args.reference_dir}")
    print(f"输出目录: {output_dir}")
    print(f"设备: {Config.DEVICE}")
    print(f"批大小: {args.batch_size} | Patch尺寸: {args.patch_size}")
    print(f"训练轮数: {args.epochs} | 学习率: {args.lr}")
    print(f"Loss权重: recon={args.recon_lambda}, reference={args.reference_lambda}, smooth={args.smooth_lambda}")
    print(f"参考值采样: {args.sample_ratio*100:.0f}% 像素")
    print(f"ROI模式: {'全图' if roi is None else f'ROI{roi}'}")
    print("="*70)


    try:
       
        dataset = PolSARDataset(
            data_dir=data_dir,
            roi=roi,
            training=True,
            patch_size=args.patch_size
        )
        H, W = dataset.H, dataset.W


        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True,
            collate_fn=polsar_collate_fn
        )
        
        print(f"✅ 数据集加载完成 | 训练集Patch数: {len(train_loader)} | ROI尺寸: {H}x{W}")
    except Exception as e:
        raise RuntimeError(f"数据集加载失败：{e}")

    # ==================== 模型初始化 ====================
    model = CNNPolDecompAE(decomp_type=args.model_type).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️ 模型初始化完成 | 参数量: {total_params/1e6:.2f}M")

    # ==================== 优化器配置 ====================
    loss_fn = get_loss_fn(
        model_type_enum,
        reference_dir=args.reference_dir,
        H=H,
        W=W,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                          weight_decay=Config.DEFAULT_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min',
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=Config.LR_MIN
    )

    # ==================== Checkpoint加载 ====================
    checkpoint_path = Config.get_checkpoint_path(model_type_str)
    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(checkpoint_path):
        try:

            import torch.version
            torch_ver = torch.version.__version__.split('.')
            if int(torch_ver[0]) >= 2 and int(torch_ver[1]) >= 0:
                ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=True)
            else:
                ckpt = torch.load(checkpoint_path, map_location=Config.DEVICE)
                
            model.load_state_dict(ckpt['model_state_dict'])
            if not args.inference:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                best_loss = ckpt.get('loss', float('inf'))
            print(f"✅ Checkpoint加载成功 | Epoch: {start_epoch} | Loss: {best_loss:.6f}")
        except Exception as e:
            print(f"⚠️ Checkpoint加载失败: {e}")

    # ==================== 训练阶段 ====================
    if not args.inference:
        print("\n=== 开始训练 ===")
        no_improve_epochs = 0

        for epoch in range(start_epoch, args.epochs):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch_idx, batch_data in enumerate(pbar):
                if len(batch_data) == 3:
                    
                    batch_input, batch_t3_raw, batch_coords = batch_data
                else:
            
                    batch_input, batch_t3_raw = batch_data
                    batch_coords = None
                
                
                batch_input = batch_input.to(Config.DEVICE, non_blocking=True)
                batch_t3_raw = batch_t3_raw.to(Config.DEVICE, non_blocking=True)
                T_target = unflatten_t3_image(batch_t3_raw)

                optimizer.zero_grad()
                T_recon, contrib, _ = model(batch_input, T_target=T_target, is_training=True)

               
                if args.model_type == '6comp' and 'T_target_derotated' in contrib:
                    T_target_for_loss = contrib['T_target_derotated']
                    assert T_target_for_loss.shape == T_target.shape, \
                        f"T_target_derotated维度{T_target_for_loss.shape}与T_target{T_target.shape}不匹配"
                else:
                    T_target_for_loss = T_target
                
                # Flatten计算Loss
                T_recon_flat = T_recon.permute(0, 3, 4, 1, 2).reshape(-1, 3, 3)
                T_target_flat = T_target_for_loss.permute(0, 3, 4, 1, 2).reshape(-1, 3, 3)

                
                loss, loss_terms = loss_fn(
                    T_pred=T_recon_flat,
                    T_target=T_target_flat,
                    contrib=contrib,
                    patch_coords=batch_coords,
                    recon_lambda=args.recon_lambda,
                    reference_lambda=args.reference_lambda,
                    smooth_lambda=args.smooth_lambda,
                    sample_ratio=args.sample_ratio,
                )

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()


                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{loss_terms.get('recon_total_recon', 0):.4f}",
                    'ref': f"{loss_terms.get('ref_total_reference', 0):.4f}",
                    'smooth': f"{loss_terms.get('smooth_tv_total', 0):.4f}"
                })

            # Epoch统计
            avg_loss = epoch_loss / len(train_loader)
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{args.epochs} 完成 | 平均Loss: {avg_loss:.6f}")
            print(f"{'='*80}")

            # 打印最后一个batch的统计
            print_batch_statistics(contrib, model_type_str, args.remove_outliers)

            scheduler.step(avg_loss)

            if avg_loss < (best_loss - Config.EARLY_MIN_DELTA):
                best_loss = avg_loss
                no_improve_epochs = 0
                
                import torch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, checkpoint_path)
                print(f"🌟 保存最佳模型 | Loss: {best_loss:.6f}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= Config.EARLY_PATIENCE:
                    print("⛔ 早停触发")
                    break

    # 推理阶段
    print("\n" + "="*70)
    print(f"开始全图推理 | 模型类型: {model_type_str}")
    print("="*70)

    model.eval()
    all_T_pred = []
    all_T_target = []
    
    BLOCK_SIZE = Config.INFERENCE_BLOCK_SIZE
    STRIDE = Config.INFERENCE_STRIDE
    H_full, W_full = dataset.H, dataset.W
    
    # 初始化存储
    contrib_full = {
        'ps': np.zeros((H_full, W_full), dtype=np.float32),
        'pd': np.zeros((H_full, W_full), dtype=np.float32),
        'pv': np.zeros((H_full, W_full), dtype=np.float32),
        'beta_real': np.zeros((H_full, W_full), dtype=np.float32),
        'beta_imag': np.zeros((H_full, W_full), dtype=np.float32),
        'alpha_real': np.zeros((H_full, W_full), dtype=np.float32),
        'alpha_imag': np.zeros((H_full, W_full), dtype=np.float32),
    }

    contrib_count = np.zeros((H_full, W_full), dtype=np.float32)

    if model_type_str in ['4comp', '6comp']:
        contrib_full.update({'ph': np.zeros((H_full, W_full), dtype=np.float32)})
    
    if model_type_str == '6comp':
        contrib_full.update({
            'pod': np.zeros((H_full, W_full), dtype=np.float32),
            'pcd': np.zeros((H_full, W_full), dtype=np.float32),
            'theta': np.zeros((H_full, W_full), dtype=np.float32),
        })

    with torch.no_grad():
        dataset.training = False
        full_data = dataset[0]
        if len(full_data) == 3:
            full_input, full_t3_raw, _ = full_data 
        else:
            full_input, full_t3_raw = full_data
            
        if full_input.dim() == 3:
            full_input = full_input.unsqueeze(0)
            full_t3_raw = full_t3_raw.unsqueeze(0)
        
        # 分块推理
        for r in tqdm(range(0, H_full, STRIDE), desc="推理中"):
            for c in range(0, W_full, STRIDE):
                r_end = min(r + BLOCK_SIZE, H_full)
                c_end = min(c + BLOCK_SIZE, W_full)
                h_block = r_end - r
                w_block = c_end - c
                
                block_input = full_input[:, :, r:r_end, c:c_end].to(Config.DEVICE)
                block_t3_raw = full_t3_raw[:, :, r:r_end, c:c_end].to(Config.DEVICE)
                
                
                mask = None
                if h_block < BLOCK_SIZE or w_block < BLOCK_SIZE:
                    pad_input = torch.zeros(1, 9, BLOCK_SIZE, BLOCK_SIZE,
                                           device=Config.DEVICE, dtype=block_input.dtype)
                    pad_t3 = torch.zeros(1, 9, BLOCK_SIZE, BLOCK_SIZE,
                                        device=Config.DEVICE, dtype=block_t3_raw.dtype)
                    pad_input[:, :, :h_block, :w_block] = block_input
                    pad_t3[:, :, :h_block, :w_block] = block_t3_raw
                    block_input = pad_input
                    block_t3_raw = pad_t3
                    # 定义有效区域掩码
                    mask = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float32, device=Config.DEVICE)
                    mask[:h_block, :w_block] = 1.0
                else:
                    mask = torch.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=torch.float32, device=Config.DEVICE)
                
                T_target_block = unflatten_t3_image(block_t3_raw)
                T_recon_block, contrib_block, _ = model(block_input, T_target=T_target_block, is_training=False)
                
     
                T_recon_flat = T_recon_block[:, :, :, :h_block, :w_block].permute(0, 3, 4, 1, 2).reshape(-1, 3, 3)
                T_target_flat = T_target_block[:, :, :, :h_block, :w_block].permute(0, 3, 4, 1, 2).reshape(-1, 3, 3)
                all_T_pred.append(T_recon_flat.cpu())
                all_T_target.append(T_target_flat.cpu())
                
          
                for k in contrib_full.keys():
                    if k in contrib_block:
                        data = contrib_block[k].squeeze(0).cpu().numpy()[:h_block, :w_block]
                        contrib_full[k][r:r_end, c:c_end] += data

                contrib_count[r:r+h_block, c:c+w_block] += 1.0


                del block_input, block_t3_raw, T_recon_block, contrib_block, T_target_block
                if 'pad_input' in locals():
                    del pad_input, pad_t3
                if mask is not None:
                    del mask
                torch.cuda.empty_cache()

        # 归一化累加结果
        for k in contrib_full.keys():
            contrib_full[k] /= np.maximum(contrib_count, 1e-6)

        print(f"\n💾 保存结果到 {output_dir}...")
        for k in contrib_full.keys():
            save_bin(
                torch.from_numpy(contrib_full[k]), 
                k, 
                output_dir, 
                model_type_str,
            )
        
        # 打印T矩阵统计
        if len(all_T_pred) > 0:
            global_T_pred = torch.cat(all_T_pred, dim=0)
            global_T_target = torch.cat(all_T_target, dim=0)
            print_final_T_values(global_T_pred, global_T_target)
            print("\n🎉 推理完成 | 结果已保存到: " + output_dir)

if __name__ == "__main__":
    main()