#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

def compare_schedulers():
    """比较不同学习率调度器的效果"""
    print("📊 学习率调度器对比分析")
    print("=" * 60)
    
    # 模拟参数
    base_lr = 0.004
    total_epochs = 100
    steps_per_epoch = 50
    total_steps = total_epochs * steps_per_epoch
    
    # 创建虚拟模型和数据
    model = nn.Linear(10, 1).double()
    
    # 不同调度器配置
    schedulers_config = {
        'ConstantLR (固定)': {
            'type': 'constant',
            'optimizer': optim.AdamW(model.parameters(), lr=base_lr),
            'scheduler': None
        },
        'StepLR (阶梯式)': {
            'type': 'step',
            'optimizer': optim.AdamW(model.parameters(), lr=base_lr),
            'scheduler': optim.lr_scheduler.StepLR(
                optim.AdamW(model.parameters(), lr=base_lr), 
                step_size=20, gamma=0.5
            )
        },
        'ReduceLROnPlateau (自适应)': {
            'type': 'plateau',
            'optimizer': optim.AdamW(model.parameters(), lr=base_lr),
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                optim.AdamW(model.parameters(), lr=base_lr),
                mode='min', factor=0.8, patience=10
            )
        },
        'OneCycleLR (余弦)': {
            'type': 'onecycle',
            'optimizer': optim.AdamW(model.parameters(), lr=base_lr),
            'scheduler': optim.lr_scheduler.OneCycleLR(
                optim.AdamW(model.parameters(), lr=base_lr),
                max_lr=base_lr * 2,
                total_steps=total_steps,
                pct_start=0.15,
                anneal_strategy='cos'
            )
        }
    }
    
    # 记录学习率变化
    lr_histories = {}
    
    for name, config in schedulers_config.items():
        optimizer = config['optimizer']
        scheduler = config['scheduler']
        lr_history = []
        
        # 模拟损失变化（用于ReduceLROnPlateau）
        losses = []
        for epoch in range(total_epochs):
            # 模拟一个先下降后平稳的损失曲线
            base_loss = 1.0 * np.exp(-epoch / 30) + 0.1
            noise = np.random.normal(0, 0.05)
            epoch_loss = max(0.05, base_loss + noise)
            losses.append(epoch_loss)
            
            for step in range(steps_per_epoch):
                lr_history.append(optimizer.param_groups[0]['lr'])
                
                if config['type'] == 'onecycle':
                    scheduler.step()
            
            # Epoch级别的调度器
            if config['type'] == 'step':
                scheduler.step()
            elif config['type'] == 'plateau':
                scheduler.step(epoch_loss)
        
        lr_histories[name] = lr_history
    
    # 分析结果
    print("\n🔍 调度器特性分析:")
    print("-" * 50)
    
    for name, lr_history in lr_histories.items():
        lr_array = np.array(lr_history)
        
        print(f"\n{name}:")
        print(f"  初始学习率: {lr_array[0]:.6f}")
        print(f"  最大学习率: {lr_array.max():.6f}")
        print(f"  最小学习率: {lr_array.min():.6f}")
        print(f"  最终学习率: {lr_array[-1]:.6f}")
        print(f"  动态范围: {lr_array.max()/lr_array.min():.1f}x")
        print(f"  平均学习率: {lr_array.mean():.6f}")
        
        # 计算学习率的变化幅度（标准差）
        lr_std = lr_array.std()
        print(f"  变化幅度: {lr_std:.6f}")
    
    return lr_histories

def explain_onecycle_advantages():
    """解释OneCycleLR的优势"""
    print("\n🎯 OneCycleLR的优势详解:")
    print("=" * 50)
    
    advantages = {
        "🚀 快速收敛": [
            "高学习率阶段快速逃离局部最优",
            "大步长探索参数空间",
            "比固定学习率快2-10倍收敛"
        ],
        
        "🎯 更好的泛化": [
            "高学习率增加噪声，提供正则化效果", 
            "余弦衰减平滑过渡到精细调优",
            "避免过拟合，提升测试性能"
        ],
        
        "🔄 自适应性": [
            "无需手动调整学习率",
            "自动平衡探索与利用",
            "对不同数据集和模型架构鲁棒"
        ],
        
        "⚡ 训练稳定性": [
            "Warmup阶段避免初期梯度爆炸",
            "余弦衰减防止后期震荡",
            "动量调节配合学习率变化"
        ]
    }
    
    for category, points in advantages.items():
        print(f"\n{category}:")
        for point in points:
            print(f"  ✅ {point}")

def demonstrate_classification_benefits():
    """演示分类任务的特殊需求"""
    print("\n🧠 分类任务的特殊需求:")
    print("=" * 40)
    
    print("""
🎯 分类 vs 回归的差异:
├── 分类: 需要清晰的决策边界
├── 回归: 需要平滑的函数拟合
└── 分类更受益于"粗调→细调"的学习过程

🔄 OneCycleLR在分类中的作用:
├── Phase 1 (Warmup): 温和启动，避免坏的初始化
├── Phase 2 (High LR): 快速找到好的区域，逃离局部最优  
├── Phase 3 (Decay): 精细调优决策边界
└── 整个过程类似"先大致定位，再精确瞄准"

🆚 与其他调度器对比:
├── 固定LR: 可能陷入局部最优，收敛慢
├── StepLR: 突然的学习率下降可能破坏训练动态
├── ReduceLROnPlateau: 被动响应，可能错过最佳时机
└── OneCycleLR: 主动设计，基于训练理论优化

💡 为什么适合复数分类:
├── 复数网络参数空间更复杂
├── 需要更强的探索能力找到好的参数
├── ComplexCosineActivation的可学习参数需要精细调优
└── 幅度+相位双重优化目标需要平衡的学习策略
    """)

if __name__ == '__main__':
    lr_histories = compare_schedulers()
    explain_onecycle_advantages()
    demonstrate_classification_benefits() 