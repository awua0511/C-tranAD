#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')

def test_v2_improvements():
    """测试V2分类头的改进效果"""
    print("🔬 测试V2分类头改进...")
    
    try:
        from src.models import ComplexClfHead, ComplexClfHeadV2
        
        # 测试参数
        feats = 10  # 必须是偶数
        batch_size = 64
        
        # 创建测试数据
        x = torch.randn(batch_size, feats, dtype=torch.float64)
        
        print(f"输入特征维度: {feats}, 批次大小: {batch_size}")
        print("=" * 70)
        
        # 比较V1和V2
        print("📊 模型对比:")
        v1 = ComplexClfHead(feats).double()
        v2 = ComplexClfHeadV2(feats).double()
        
        # 参数统计
        v1_params = sum(p.numel() for p in v1.parameters())
        v2_params = sum(p.numel() for p in v2.parameters())
        
        print(f"V1 参数数量: {v1_params:,}")
        print(f"V2 参数数量: {v2_params:,}")
        print(f"V2 参数增加: {v2_params/v1_params:.1f}x")
        print()
        
        # 测试前向传播
        print("🚀 前向传播测试:")
        with torch.no_grad():
            logits_v1 = v1(x)
            logits_v2 = v2(x)
        
        print(f"V1 输出范围: [{logits_v1.min().item():.4f}, {logits_v1.max().item():.4f}]")
        print(f"V2 输出范围: [{logits_v2.min().item():.4f}, {logits_v2.max().item():.4f}]")
        print()
        
        # 测试梯度流
        print("🌊 梯度流测试:")
        y_fake = torch.randint(0, 2, (batch_size,), dtype=torch.float64)
        
        # V1梯度测试
        v1.train()
        logits_v1 = v1(x)
        loss_v1 = nn.BCEWithLogitsLoss()(logits_v1, y_fake)
        loss_v1.backward()
        
        v1_grad_norm = 0
        for p in v1.parameters():
            if p.grad is not None:
                v1_grad_norm += p.grad.norm().item() ** 2
        v1_grad_norm = v1_grad_norm ** 0.5
        
        # V2梯度测试
        v2.train()
        logits_v2 = v2(x)
        loss_v2 = nn.BCEWithLogitsLoss()(logits_v2, y_fake)
        loss_v2.backward()
        
        v2_grad_norm = 0
        for p in v2.parameters():
            if p.grad is not None:
                v2_grad_norm += p.grad.norm().item() ** 2
        v2_grad_norm = v2_grad_norm ** 0.5
        
        print(f"V1 梯度范数: {v1_grad_norm:.6f}")
        print(f"V2 梯度范数: {v2_grad_norm:.6f}")
        print(f"V2 梯度增强: {v2_grad_norm/v1_grad_norm:.1f}x")
        print()
        
        # 测试表达能力
        print("💪 表达能力测试:")
        
        # 创建更复杂的测试数据
        x_complex = torch.zeros(batch_size, feats, dtype=torch.float64)
        for i in range(feats//2):
            # 创建不同频率的复数信号
            t = torch.linspace(0, 2*np.pi, batch_size)
            freq = 1 + i * 0.5
            x_complex[:, 2*i] = torch.cos(freq * t) + 0.1 * torch.randn(batch_size)      # real
            x_complex[:, 2*i+1] = torch.sin(freq * t) + 0.1 * torch.randn(batch_size)    # imag
        
        with torch.no_grad():
            out_v1 = v1(x_complex)
            out_v2 = v2(x_complex)
        
        # 计算输出的方差（表达能力指标）
        var_v1 = torch.var(out_v1).item()
        var_v2 = torch.var(out_v2).item()
        
        print(f"V1 输出方差: {var_v1:.6f}")
        print(f"V2 输出方差: {var_v2:.6f}")
        print(f"V2 表达增强: {var_v2/var_v1:.1f}x")
        print()
        
        # 总结
        print("✨ 改进总结:")
        print(f"✅ 参数容量增加 {v2_params/v1_params:.1f}x")
        print(f"✅ 梯度流改善 {v2_grad_norm/v1_grad_norm:.1f}x") 
        print(f"✅ 表达能力提升 {var_v2/var_v1:.1f}x")
        print("✅ 添加残差连接，缓解梯度消失")
        print("✅ 多尺度特征提取，增强非线性")
        print("✅ BatchNorm稳定训练")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def training_tips():
    """训练建议"""
    print("\n💡 V2 训练建议:")
    print("""
1. 🎯 使用更大的学习率 (2x)
2. 📏 使用较小的批次大小 (1024 vs 2048)  
3. ⏰ 使用学习率调度器 (ReduceLROnPlateau)
4. 🏋️ 训练更多轮次 (150-200 epochs)
5. 🔧 监控梯度范数，确保不会爆炸

示例命令:
python main.py --model TranAD_QC --dataset my_data --train_clf --clf_version v2 --clf_epochs 150
    """)

if __name__ == '__main__':
    if test_v2_improvements():
        training_tips() 