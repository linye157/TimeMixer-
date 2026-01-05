#!/usr/bin/env python
"""
Inspect intermediate tensor shapes in TimeMixer++ model.

This script helps visualize the data flow through each module:
- Multi-scale generation
- MRTI (Multi-Resolution Time Imaging)
- TID (Time Image Decomposition)
- MCM (Multi-Scale Mixing)
- MRM (Multi-Resolution Mixing)
- Output head

Usage:
    python scripts/inspect_shapes.py
    python scripts/inspect_shapes.py --batch_size 4 --d_model 64
    python scripts/inspect_shapes.py --checkpoint checkpoints/best_model.pt
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import torch

from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls
from timemixerpp.mrti import MRTI
from timemixerpp.tid import TID
from timemixerpp.mcm import MCM
from timemixerpp.mrm import MRM


def parse_args():
    parser = argparse.ArgumentParser(description='Inspect TimeMixer++ intermediate shapes')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for inspection')
    parser.add_argument('--seq_len', type=int, default=48, help='Sequence length')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of MixerBlock layers')
    parser.add_argument('--top_k', type=int, default=3, help='Top-K frequencies')
    parser.add_argument('--checkpoint', type=str, default=None, help='Load config from checkpoint')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    return parser.parse_args()


def print_separator(title: str, char: str = "=", width: int = 70):
    """Print a section separator."""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")


def print_tensor_info(name: str, tensor: torch.Tensor, indent: int = 2):
    """Print tensor shape information."""
    prefix = " " * indent
    shape_str = " × ".join(map(str, tensor.shape))
    print(f"{prefix}{name}: ({shape_str})")


def inspect_model_shapes(config: TimeMixerPPConfig, batch_size: int, device: str):
    """Inspect all intermediate shapes in the model."""
    
    print_separator("TimeMixer++ 中间形状检查", "=")
    print(f"\n配置参数:")
    print(f"  batch_size (B) = {batch_size}")
    print(f"  seq_len (T) = {config.seq_len}")
    print(f"  d_model = {config.d_model}")
    print(f"  n_layers = {config.n_layers}")
    print(f"  top_k (K) = {config.top_k}")
    print(f"  动态尺度数 M = {config.compute_dynamic_M()}")
    print(f"  各尺度长度 = {config.get_scale_lengths()}")
    
    # Create model
    model = TimeMixerPPForBinaryCls(config).to(device)
    model.eval()
    
    B = batch_size
    T = config.seq_len
    d = config.d_model
    M = config.compute_dynamic_M()
    lengths = config.get_scale_lengths()
    
    # ========================================
    # Step 1: Input
    # ========================================
    print_separator("1. 输入", "-")
    x = torch.randn(B, T, device=device)
    print_tensor_info("原始输入 x", x)
    print(f"    说明: B={B} 个样本，每个样本 T={T} 个时间步")
    
    # ========================================
    # Step 2: Input Projection
    # ========================================
    print_separator("2. 输入投影 (Input Projection)", "-")
    x_proj = model.encoder.input_proj(x.unsqueeze(-1))
    print_tensor_info("投影后 x_proj", x_proj)
    print(f"    说明: 从 c_in=1 投影到 d_model={d}")
    
    # ========================================
    # Step 3: Multi-scale Generation
    # ========================================
    print_separator("3. 多尺度生成 (Multi-Scale Generation)", "-")
    multi_scale_x = model.encoder.multi_scale_gen(x_proj)
    print(f"  生成 M+1 = {len(multi_scale_x)} 个尺度:")
    for m, x_m in enumerate(multi_scale_x):
        print_tensor_info(f"x_{m} (尺度 {m}, L_{m}={lengths[m]})", x_m, indent=4)
    print(f"\n    说明: 每个尺度通过 stride=2 的 Conv1d 下采样")
    print(f"    L_m = T / 2^m = {T} / 2^m")
    
    # ========================================
    # Step 4: Inside MixerBlock - MRTI
    # ========================================
    print_separator("4. MRTI (多分辨率时间成像)", "-")
    
    block = model.encoder.blocks[0]
    mrti = block.mrti
    
    time_images, periods, amplitudes = mrti(multi_scale_x)
    K_eff = len(periods)
    
    print(f"  检测到的周期 (K_eff={K_eff}): {periods}")
    print_tensor_info("幅值权重 amplitudes", amplitudes)
    print(f"\n  对于每个周期 k，将各尺度的 1D 序列重塑为 2D 时间图像:")
    
    for k, ti in enumerate(time_images):
        print(f"\n  周期 k={k}, period={ti.period}:")
        for m, img in enumerate(ti.images):
            H = ti.period
            W = img.shape[3]
            print(f"      z_{m}^({k}): (B={B}, d={d}, H={H}, W={W})")
            print(f"          H=period={H}, W=ceil(L_{m}/period)=ceil({lengths[m]}/{H})={W}")
    
    print(f"\n    说明: 1D→2D 重塑公式")
    print(f"    (B, L_m, d) → pad → (B, d, period, ceil(L_m/period))")
    
    # ========================================
    # Step 5: Inside MixerBlock - TID
    # ========================================
    print_separator("5. TID (时间图像分解)", "-")
    
    tid = block.tid
    
    print(f"  对每个周期的时间图像进行双轴注意力分解:")
    print(f"  - 列注意力 (Column Attention) → 季节性分量 s")
    print(f"  - 行注意力 (Row Attention) → 趋势分量 t")
    
    for k, ti in enumerate(time_images[:1]):  # Only show first period for brevity
        seasonal_imgs, trend_imgs = tid(ti.images)
        print(f"\n  周期 k={k} 的分解结果:")
        for m, (s, t) in enumerate(zip(seasonal_imgs, trend_imgs)):
            print(f"      尺度 {m}:")
            print(f"        季节性 s_{m}^({k}): {tuple(s.shape)}")
            print(f"        趋势   t_{m}^({k}): {tuple(t.shape)}")
    
    print(f"\n    说明: TID 保持形状不变")
    print(f"    输入 z: (B, d, H, W) → 输出 s, t: 各 (B, d, H, W)")
    print(f"\n    列注意力: (B, d, H, W) → (B*H, W, d) → MHSA → (B, d, H, W)")
    print(f"    行注意力: (B, d, H, W) → (B*W, H, d) → MHSA → (B, d, H, W)")
    
    # ========================================
    # Step 6: Inside MixerBlock - MCM
    # ========================================
    print_separator("6. MCM (多尺度混合)", "-")
    
    mcm = block.mcm
    
    print(f"  对每个周期 k，进行跨尺度混合:")
    print(f"  - 季节性: 自底向上 (Bottom-Up) s_m += ConvDown(s_{{m-1}})")
    print(f"  - 趋势:   自顶向下 (Top-Down)  t_m += TransConvUp(t_{{m+1}})")
    
    for k, ti in enumerate(time_images[:1]):
        seasonal_imgs, trend_imgs = tid(ti.images)
        z_list = mcm(seasonal_imgs, trend_imgs, ti.original_lengths, ti.period)
        
        print(f"\n  周期 k={k} 混合后，2D→1D 还原:")
        for m, z in enumerate(z_list):
            print(f"      z_{m}^({k}): {tuple(z.shape)} (原始长度 L_{m}={ti.original_lengths[m]})")
    
    print(f"\n    说明: MCM 后将 2D 图像还原为 1D 序列并截断到原始长度")
    print(f"    (B, d, H, W) → (B, H*W, d) → 截断 → (B, L_m, d)")
    
    # ========================================
    # Step 7: Inside MixerBlock - MRM
    # ========================================
    print_separator("7. MRM (多分辨率混合)", "-")
    
    # Collect all periods' outputs
    z_per_period = []
    for ti in time_images:
        seasonal_imgs, trend_imgs = tid(ti.images)
        z_list = mcm(seasonal_imgs, trend_imgs, ti.original_lengths, ti.period)
        z_per_period.append(z_list)
    
    mrm = block.mrm
    x_out = mrm(z_per_period, amplitudes)
    
    print(f"  将 K_eff={K_eff} 个周期的结果进行幅值加权聚合:")
    print(f"  x_m = Σ_k softmax(A)[k] × z_m^(k)")
    print(f"\n  聚合后各尺度输出:")
    for m, x_m in enumerate(x_out):
        print_tensor_info(f"x_{m}^{{out}}", x_m, indent=4)
    
    print(f"\n    说明: 对每个尺度 m，跨周期 k 加权求和")
    print(f"    权重来自 FFT 幅值的 softmax 归一化")
    
    # ========================================
    # Step 8: After all MixerBlocks
    # ========================================
    print_separator("8. 经过所有 MixerBlock 后", "-")
    
    # Run through all blocks
    current = multi_scale_x
    for layer_idx, block in enumerate(model.encoder.blocks):
        current = block(current)
        print(f"\n  Layer {layer_idx + 1} 输出:")
        for m, x_m in enumerate(current):
            print_tensor_info(f"x_{m}^{{L{layer_idx+1}}}", x_m, indent=4)
    
    # ========================================
    # Step 9: Output Head
    # ========================================
    print_separator("9. 输出头 (Output Head)", "-")
    
    print(f"  对每个尺度进行池化 + 线性投影:")
    for m, x_m in enumerate(current):
        pooled = x_m.mean(dim=1)  # Mean pooling
        print(f"    尺度 {m}: {tuple(x_m.shape)} → 池化 → {tuple(pooled.shape)} → Linear → (B, 1)")
    
    # Final output
    with torch.no_grad():
        output = model(x)
    
    print(f"\n  多尺度集成后:")
    print_tensor_info("logits", output['logits'])
    print_tensor_info("probs", output['probs'])
    
    print_separator("形状检查完成", "=")
    
    # Summary table
    print("\n📊 形状变化总结表:\n")
    print("| 阶段 | 输入形状 | 输出形状 | 说明 |")
    print("|------|----------|----------|------|")
    print(f"| 输入 | (B, T) | (B, T, 1) | 增加通道维度 |")
    print(f"| 投影 | (B, T, 1) | (B, T, d) | Linear: 1→{d} |")
    print(f"| 多尺度 | (B, T, d) | [(B, L_m, d)]×(M+1) | Conv1d stride=2 |")
    print(f"| MRTI | (B, L_m, d) | (B, d, H, W) | 1D→2D, H=period |")
    print(f"| TID | (B, d, H, W) | s,t: (B, d, H, W) | 双轴注意力 |")
    print(f"| MCM | s,t: (B, d, H, W) | (B, L_m, d) | 2D→1D |")
    print(f"| MRM | [(B, L_m, d)]×K | (B, L_m, d) | 跨周期聚合 |")
    print(f"| 输出头 | [(B, L_m, d)]×(M+1) | (B, 1) | 池化+集成 |")
    print()


def main():
    args = parse_args()
    
    # Load config from checkpoint or create new
    if args.checkpoint:
        import torch as th
        checkpoint = th.load(args.checkpoint, map_location='cpu')
        if 'config' in checkpoint:
            config = TimeMixerPPConfig(**checkpoint['config'])
            print(f"从检查点加载配置: {args.checkpoint}")
        else:
            config = TimeMixerPPConfig(
                seq_len=args.seq_len,
                d_model=args.d_model,
                n_layers=args.n_layers,
                top_k=args.top_k
            )
    else:
        config = TimeMixerPPConfig(
            seq_len=args.seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            top_k=args.top_k
        )
    
    inspect_model_shapes(config, args.batch_size, args.device)


if __name__ == '__main__':
    main()

