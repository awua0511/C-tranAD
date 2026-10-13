#!/usr/bin/env python3
"""
分类结果可视化示例脚本
演示如何使用新的可视化功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置环境变量以避免图形界面问题
os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互式后端

import numpy as np
import torch
from src.models import TranAD_QC
from visualization import ClassificationVisualizer

def create_dummy_data():
    """创建示例数据用于演示"""
    print("创建示例数据...")
    
    # 模拟参数
    n_windows = 1000
    window_size = 10
    n_features = 8  # 必须是偶数，用于复数
    
    # 创建正常和异常数据
    np.random.seed(42)
    
    # 正常数据：较小的随机波动
    normal_windows = np.random.normal(0, 0.5, (800, window_size, n_features))
    
    # 异常数据：较大的波动和偏移
    anomaly_windows = np.random.normal(2, 1.5, (200, window_size, n_features))
    
    # 合并数据
    test_windows = np.concatenate([normal_windows, anomaly_windows], axis=0)
    
    # 创建标签
    test_labels = np.zeros((n_windows, n_features))
    test_labels[800:, :] = 1  # 最后200个为异常
    
    # 随机打乱
    indices = np.random.permutation(n_windows)
    test_windows = test_windows[indices]
    test_labels = test_labels[indices]
    
    print(f"数据形状: test_windows={test_windows.shape}, test_labels={test_labels.shape}")
    print(f"异常比例: {(test_labels.sum(axis=1) > 0).mean():.2%}")
    
    return test_windows, test_labels

def create_dummy_model():
    """创建一个示例模型"""
    print("创建示例模型...")
    
    # 创建TranAD_QC模型
    model = TranAD_QC(
        feats=8, 
        activation_type='learnable_cosine',
        loss_type='complex_aware',
        focus_type='complex_aware'
    )
    model = model.double()
    model.eval()
    
    print(f"模型名称: {model.name}")
    return model

def demo_basic_visualization():
    """演示基础可视化功能"""
    print("\n" + "="*60)
    print("演示基础分类结果可视化功能")
    print("="*60)
    
    # 创建数据和模型
    test_windows, test_labels = create_dummy_data()
    model = create_dummy_model()
    
    # 创建可视化器
    visualizer = ClassificationVisualizer(save_dir="plots/demo")
    
    try:
        # 提取特征和标签
        print("\n1. 提取特征...")
        features, scores, labels = visualizer.extract_features_and_labels(
            model, test_windows, test_labels, mode='complex')
        print(f"特征形状: {features.shape}")
        print(f"分数形状: {scores.shape}")
        print(f"标签形状: {labels.shape}")
        
        # 降维
        print("\n2. 降维到2D...")
        features_2d_pca = visualizer.reduce_dimensions(features, method='pca')
        print(f"PCA降维后形状: {features_2d_pca.shape}")
        
        # 生成分类边界图
        print("\n3. 生成分类边界可视化...")
        visualizer.plot_classification_boundary(
            features_2d_pca, labels, scores,
            title="示例：分类边界可视化",
            save_name="demo_classification_boundary.png"
        )
        
        # 生成混淆矩阵
        print("\n4. 生成混淆矩阵...")
        predictions = (scores > 0).astype(int)
        visualizer.plot_confusion_matrix(
            labels, predictions,
            title="示例：混淆矩阵",
            save_name="demo_confusion_matrix.png"
        )
        
        # 生成ROC曲线
        print("\n5. 生成ROC曲线...")
        auc_score = visualizer.plot_roc_curve(
            labels, scores,
            title="示例：ROC曲线",
            save_name="demo_roc_curve.png"
        )
        print(f"AUC分数: {auc_score:.4f}")
        
        print(f"\n✓ 基础可视化演示完成！结果保存在 plots/demo/ 目录")
        
    except Exception as e:
        print(f"✗ 基础可视化演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_multi_fold_visualization():
    """演示多fold可视化功能"""
    print("\n" + "="*60)
    print("演示多fold交叉验证可视化功能")
    print("="*60)
    
    # 创建数据和模型
    test_windows, test_labels = create_dummy_data()
    model = create_dummy_model()
    
    # 创建可视化器
    visualizer = ClassificationVisualizer(save_dir="plots/demo_multifold")
    
    try:
        # 进行多fold可视化
        print("\n进行3折交叉验证可视化...")
        fold_results = visualizer.multi_fold_visualization(
            model, test_windows, test_labels, n_folds=3, mode='complex')
        
        print(f"\n✓ 多fold可视化演示完成！结果保存在 plots/demo_multifold/ 目录")
        print(f"总共处理了 {len(fold_results)} 个fold")
        
    except Exception as e:
        print(f"✗ 多fold可视化演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_full_report():
    """演示完整报告生成"""
    print("\n" + "="*60)
    print("演示完整可视化报告生成")
    print("="*60)
    
    # 创建数据和模型
    test_windows, test_labels = create_dummy_data()
    model = create_dummy_model()
    
    # 创建可视化器
    visualizer = ClassificationVisualizer(save_dir="plots/demo_full")
    
    try:
        # 生成完整报告
        print("\n生成完整可视化报告...")
        visualizer.generate_full_report(
            model, test_windows, test_labels, 
            model_name=model.name, n_folds=3
        )
        
        print(f"\n✓ 完整报告演示完成！")
        
    except Exception as e:
        print(f"✗ 完整报告演示失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主演示函数"""
    print("分类结果可视化功能演示")
    print("=" * 60)
    
    # 运行各种演示
    demo_basic_visualization()
    demo_multi_fold_visualization() 
    demo_full_report()
    
    print("\n" + "="*60)
    print("所有演示完成！")
    print("可以查看以下目录的结果：")
    print("- plots/demo/ : 基础可视化结果")
    print("- plots/demo_multifold/ : 多fold可视化结果") 
    print("- plots/demo_full/ : 完整报告结果")
    print("="*60)

if __name__ == "__main__":
    main()


