#!/usr/bin/env python3
"""
测试F1-score可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
from visualization import ClassificationVisualizer

def create_dummy_data():
    """创建虚拟数据用于测试"""
    np.random.seed(42)
    
    # 创建1000个样本
    n_samples = 1000
    n_features = 10
    
    # 生成特征
    features = np.random.randn(n_samples, n_features)
    
    # 生成标签（80%正常，20%异常）
    labels = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # 生成分数（基于标签和噪声）
    scores = np.random.randn(n_samples) + labels * 2  # 异常样本分数更高
    
    return features, scores, labels

def test_f1_visualization():
    """测试F1-score可视化功能"""
    print("测试F1-score可视化功能")
    print("="*50)
    
    # 创建测试数据
    features, scores, labels = create_dummy_data()
    
    # 创建可视化器
    visualizer = ClassificationVisualizer(save_dir="plots/f1_test")
    
    print(f"数据信息:")
    print(f"- 样本数量: {len(labels)}")
    print(f"- 正常样本: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)")
    print(f"- 异常样本: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)")
    print(f"- 分数范围: [{scores.min():.2f}, {scores.max():.2f}]")
    
    # 测试F1-score分析
    print("\n1. 生成F1-score分析图...")
    f1_results = visualizer.plot_f1_score_analysis(
        labels, scores,
        title="测试 - F1-Score Analysis",
        save_name="test_f1_analysis.png"
    )
    
    print(f"最佳F1-score: {f1_results['best_f1_score']:.3f}")
    print(f"最佳阈值: {f1_results['best_threshold']:.3f}")
    
    # 测试F1-score比较（如果有多个模型结果）
    print("\n2. 生成F1-score比较图...")
    
    # 模拟多个模型的结果
    results_dict = {
        'Model_A': {'f1_score': 0.85, 'precision': 0.82, 'recall': 0.88, 'accuracy': 0.90},
        'Model_B': {'f1_score': 0.78, 'precision': 0.75, 'recall': 0.81, 'accuracy': 0.87},
        'Model_C': {'f1_score': 0.92, 'precision': 0.89, 'recall': 0.95, 'accuracy': 0.94},
        'Model_D': {'f1_score': 0.73, 'precision': 0.70, 'recall': 0.76, 'accuracy': 0.84}
    }
    
    visualizer.plot_f1_comparison(
        results_dict,
        title="测试 - F1-Score Comparison",
        save_name="test_f1_comparison.png"
    )
    
    print("\n✓ F1-score可视化测试完成！")
    print("结果保存在 plots/f1_test/ 目录")

if __name__ == "__main__":
    test_f1_visualization()
