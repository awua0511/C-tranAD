#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')

def test_all_classifier_versions():
    """测试所有版本分类头的改进效果"""
    print("🧪 测试所有分类头版本...")
    
    try:
        from src.models import ComplexClfHead, ComplexClfHeadV2, ComplexClfHeadV3
        
        # 测试参数
        feats = 10  # 必须是偶数
        batch_size = 64
        
        # 创建测试数据
        x = torch.randn(batch_size, feats, dtype=torch.float64)
        
        print(f"输入特征维度: {feats}, 批次大小: {batch_size}")
        print("=" * 80)
        
        # 测试所有版本
        versions = {
            'V1 (增强版)': ComplexClfHead(feats),
            'V2 (多尺度)': ComplexClfHeadV2(feats),
            'V3 (注意力)': ComplexClfHeadV3(feats)
        }
        
        results = {}
        
        for name, clf in versions.items():
            clf = clf.double()
            
            # 参数统计
            param_count = sum(p.numel() for p in clf.parameters())
            
            # 前向传播测试
            with torch.no_grad():
                logits = clf(x)
            
            # 梯度测试
            clf.train()
            y_fake = torch.randint(0, 2, (batch_size,), dtype=torch.float64)
            logits_grad = clf(x)
            loss = nn.BCEWithLogitsLoss()(logits_grad, y_fake)
            loss.backward()
            
            # 计算梯度范数
            grad_norm = 0
            for p in clf.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # 保存结果
            results[name] = {
                'params': param_count,
                'output_range': [logits.min().item(), logits.max().item()],
                'output_var': torch.var(logits).item(),
                'grad_norm': grad_norm,
                'loss': loss.item()
            }
            
            print(f"\n📊 {name}:")
            print(f"  参数数量: {param_count:,}")
            print(f"  输出范围: [{results[name]['output_range'][0]:.4f}, {results[name]['output_range'][1]:.4f}]")
            print(f"  输出方差: {results[name]['output_var']:.6f}")
            print(f"  梯度范数: {grad_norm:.6f}")
            print(f"  训练损失: {loss.item():.6f}")
        
        # 对比分析
        print(f"\n🔍 性能对比 (以V1为基准):")
        print("-" * 60)
        v1_results = results['V1 (增强版)']
        
        for name, result in results.items():
            if name == 'V1 (增强版)':
                continue
            
            param_ratio = result['params'] / v1_results['params']
            var_ratio = result['output_var'] / v1_results['output_var']
            grad_ratio = result['grad_norm'] / v1_results['grad_norm']
            
            print(f"\n{name} vs V1:")
            print(f"  参数倍数: {param_ratio:.1f}x")
            print(f"  表达能力: {var_ratio:.1f}x")
            print(f"  梯度强度: {grad_ratio:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_warmup_effectiveness():
    """测试warmup机制的有效性"""
    print("\n🔥 测试Warmup机制...")
    
    try:
        # 模拟OneCycleLR的学习率变化
        base_lr = 0.004
        max_lr = base_lr * 2
        total_steps = 1000
        warmup_ratio = 0.15  # 15% warmup
        
        # 创建调度器
        dummy_model = nn.Linear(10, 1).double()
        optimizer = torch.optim.AdamW(dummy_model.parameters(), lr=base_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )
        
        # 记录学习率变化
        lr_history = []
        for step in range(total_steps):
            lr_history.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # 分析关键阶段
        warmup_end = int(total_steps * warmup_ratio)
        
        print(f"学习率调度分析:")
        print(f"  初始学习率: {lr_history[0]:.6f}")
        print(f"  Warmup结束 (step {warmup_end}): {lr_history[warmup_end]:.6f}")
        print(f"  峰值学习率: {max(lr_history):.6f}")
        print(f"  最终学习率: {lr_history[-1]:.6f}")
        print(f"  学习率范围: {max(lr_history)/min(lr_history):.1f}x")
        
        # 计算不同阶段的平均学习率
        warmup_avg = np.mean(lr_history[:warmup_end])
        main_avg = np.mean(lr_history[warmup_end:])
        
        print(f"  Warmup阶段平均lr: {warmup_avg:.6f}")
        print(f"  主训练阶段平均lr: {main_avg:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Warmup测试失败: {e}")
        return False

def training_recommendations():
    """训练建议"""
    print("\n💡 分类头训练建议:")
    print("""
🎯 选择指南:
├── V1 (增强版): 平衡的选择，参数适中，性能良好
├── V2 (多尺度): 推荐选择，多尺度特征提取，表达能力强  
└── V3 (注意力): 最强版本，适合复杂异常模式

🔥 训练策略:
├── Warmup: 15-20% 的训练步数用于warmup
├── 学习率: OneCycleLR调度器，峰值lr=2×基础lr
├── 梯度裁剪: max_norm=1.0，防止梯度爆炸
└── 批次大小: V1=1500, V2=1024, V3=1536

⚡ 使用命令:
# V1增强版 (推荐新手)
python main.py --model TranAD_QC --dataset my_data --train_clf --clf_epochs 100

# V2多尺度版 (推荐)  
python main.py --model TranAD_QC --dataset my_data --train_clf --clf_version v2 --clf_epochs 150

# V3注意力版 (高级)
python main.py --model TranAD_QC --dataset my_data --train_clf --clf_version v3 --clf_epochs 200
    """)

if __name__ == '__main__':
    success = test_all_classifier_versions()
    if success:
        test_warmup_effectiveness()
        training_recommendations() 