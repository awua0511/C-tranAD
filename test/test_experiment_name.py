#!/usr/bin/env python3
"""
测试实验名称参数功能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.parser import args
from main import get_experiment_name

def test_experiment_name_generation():
    """测试实验名称生成功能"""
    print("测试实验名称生成功能")
    print("="*50)
    
    # 模拟不同的参数配置
    test_cases = [
        {
            'name': '默认配置',
            'model': 'TranAD_QC',
            'dataset': 'my_data',
            'experiment_name': None,
            'activation_type': 'learnable_cosine',
            'loss_type': 'complex_aware',
            'focus_type': 'complex_aware'
        },
        {
            'name': '自定义实验名称',
            'model': 'TranAD_QC',
            'dataset': 'my_data',
            'experiment_name': 'my_custom_experiment',
            'activation_type': 'learnable_cosine',
            'loss_type': 'complex_aware',
            'focus_type': 'complex_aware'
        },
        {
            'name': 'ReLU激活函数',
            'model': 'TranAD_QC',
            'dataset': 'my_data',
            'experiment_name': None,
            'activation_type': 'relu',
            'loss_type': 'complex_aware',
            'focus_type': 'complex_aware'
        },
        {
            'name': 'MSE损失函数',
            'model': 'TranAD_QC',
            'dataset': 'my_data',
            'experiment_name': None,
            'activation_type': 'learnable_cosine',
            'loss_type': 'mse',
            'focus_type': 'complex_aware'
        },
        {
            'name': '组合配置',
            'model': 'TranAD_QC',
            'dataset': 'my_data',
            'experiment_name': None,
            'activation_type': 'relu',
            'loss_type': 'mse',
            'focus_type': 'mse'
        },
        {
            'name': 'TranAD_C模型',
            'model': 'TranAD_C',
            'dataset': 'MBA',
            'experiment_name': None,
            'activation_type': 'learnable_cosine',
            'loss_type': 'complex_aware',
            'focus_type': 'complex_aware'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   模型: {test_case['model']}")
        print(f"   数据集: {test_case['dataset']}")
        print(f"   激活函数: {test_case['activation_type']}")
        print(f"   损失函数: {test_case['loss_type']}")
        print(f"   焦点类型: {test_case['focus_type']}")
        
        # 创建模拟的args对象
        class MockArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        mock_args = MockArgs(**test_case)
        
        # 生成实验名称
        experiment_name = get_experiment_name(
            test_case['model'], 
            test_case['dataset'], 
            mock_args
        )
        
        print(f"   生成的实验名称: {experiment_name}")
        
        # 显示相关的文件路径
        checkpoint_path = f"checkpoints/{experiment_name}/model.ckpt"
        vis_path = f"plots/classification/{experiment_name}/"
        
        print(f"   Checkpoint路径: {checkpoint_path}")
        print(f"   可视化路径: {vis_path}")

def test_file_path_generation():
    """测试文件路径生成"""
    print(f"\n{'='*50}")
    print("测试文件路径生成")
    print("="*50)
    
    # 测试不同的实验名称
    test_names = [
        "baseline",
        "relu_activation", 
        "mse_loss",
        "relu_mse_combo",
        "my_custom_experiment",
        "experiment_with_underscores",
        "test-2024-01-15"
    ]
    
    for name in test_names:
        print(f"\n实验名称: {name}")
        print(f"  Checkpoint: checkpoints/{name}/model.ckpt")
        print(f"  可视化目录: plots/classification/{name}/")
        print(f"  PCA分类边界: plots/classification/{name}/classification_boundary_pca.png")
        print(f"  F1-score分析: plots/classification/{name}/f1_score_analysis.png")
        print(f"  混淆矩阵: plots/classification/{name}/confusion_matrix.png")

if __name__ == "__main__":
    test_experiment_name_generation()
    test_file_path_generation()
    
    print(f"\n{'='*50}")
    print("测试完成！")
    print("="*50)
    print("\n使用方法示例:")
    print("python main.py --model TranAD_QC --dataset my_data --experiment_name 'my_experiment' --visualize")
    print("python main.py --model TranAD_QC --dataset my_data --experiment_name 'relu_test' --activation_type relu --visualize")
