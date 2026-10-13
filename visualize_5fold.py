#!/usr/bin/env python3
"""
5-Fold交叉验证过程可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import StratifiedKFold

def visualize_5fold_process():
    """可视化5-fold交叉验证的过程"""
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 100
    n_normal = 80
    n_anomaly = 20
    
    # 生成标签
    labels = np.concatenate([
        np.zeros(n_normal),      # 正常样本 (0)
        np.ones(n_anomaly)       # 异常样本 (1)
    ])
    
    # 创建5-fold分割
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('5-Fold Cross Validation Process', fontsize=16, fontweight='bold')
    
    # 1. 原始数据分布
    ax1 = axes[0, 0]
    colors = ['lightblue' if label == 0 else 'red' for label in labels]
    ax1.scatter(range(len(labels)), [0]*len(labels), c=colors, s=50, alpha=0.7)
    ax1.set_title('Original Data Distribution\n(Blue: Normal, Red: Anomaly)', fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('')
    ax1.set_ylim(-0.5, 0.5)
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    ax1.text(0.02, 0.95, f'Total: {len(labels)}\nNormal: {n_normal} (80%)\nAnomaly: {n_anomaly} (20%)', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2-6. 每个fold的划分
    fold_axes = [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]]
    fold_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(labels, labels)):
        ax = fold_axes[fold]
        
        # 绘制训练集和测试集
        train_colors = ['lightgreen' if labels[i] == 0 else 'darkgreen' for i in train_idx]
        test_colors = ['lightcoral' if labels[i] == 0 else 'darkred' for i in test_idx]
        
        # 训练集
        ax.scatter(train_idx, [0.2]*len(train_idx), c=train_colors, s=30, alpha=0.7, label='Train')
        # 测试集
        ax.scatter(test_idx, [-0.2]*len(test_idx), c=test_colors, s=30, alpha=0.7, label='Test')
        
        ax.set_title(f'Fold {fold+1}', fontweight='bold', color=fold_colors[fold])
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('')
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        # 添加统计信息
        train_normal = sum(1 for i in train_idx if labels[i] == 0)
        train_anomaly = sum(1 for i in train_idx if labels[i] == 1)
        test_normal = sum(1 for i in test_idx if labels[i] == 0)
        test_anomaly = sum(1 for i in test_idx if labels[i] == 1)
        
        stats_text = f'Train: {len(train_idx)} ({train_normal}N, {train_anomaly}A)\nTest: {len(test_idx)} ({test_normal}N, {test_anomaly}A)'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=8, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('5fold_process_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("5-fold交叉验证过程可视化已保存为: 5fold_process_visualization.png")

def visualize_fold_performance():
    """可视化每个fold的性能指标"""
    
    # 模拟5个fold的性能数据
    np.random.seed(42)
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    
    # 模拟性能指标（添加一些随机性）
    base_f1 = 0.85
    f1_scores = [base_f1 + np.random.normal(0, 0.05) for _ in range(5)]
    f1_scores = np.clip(f1_scores, 0, 1)  # 确保在0-1范围内
    
    base_acc = 0.90
    acc_scores = [base_acc + np.random.normal(0, 0.03) for _ in range(5)]
    acc_scores = np.clip(acc_scores, 0, 1)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1-Score
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
    bars1 = ax1.bar(folds, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score by Fold', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars1, f1_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 添加平均线
    mean_f1 = np.mean(f1_scores)
    ax1.axhline(y=mean_f1, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_f1:.3f}')
    ax1.legend()
    
    # Accuracy
    bars2 = ax2.bar(folds, acc_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy by Fold', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, score in zip(bars2, acc_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 添加平均线
    mean_acc = np.mean(acc_scores)
    ax2.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'Mean: {mean_acc:.3f}')
    ax2.legend()
    
    plt.suptitle('5-Fold Cross Validation Performance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('5fold_performance_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("5-fold性能可视化已保存为: 5fold_performance_visualization.png")
    
    # 打印统计信息
    print(f"\n性能统计:")
    print(f"F1-Score: 平均={mean_f1:.3f}, 标准差={np.std(f1_scores):.3f}")
    print(f"Accuracy: 平均={mean_acc:.3f}, 标准差={np.std(acc_scores):.3f}")

if __name__ == "__main__":
    print("生成5-fold交叉验证可视化...")
    visualize_5fold_process()
    print("\n生成5-fold性能可视化...")
    visualize_fold_performance()
    print("\n✓ 所有可视化完成！")
