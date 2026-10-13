#!/usr/bin/env python3
"""
测试模型名称修复是否正确
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models import TranAD_QC

def test_model_name_matching():
    """测试模型名称匹配逻辑"""
    print("=== 测试模型名称匹配逻辑 ===")
    
    # 测试不同的模型名称
    test_names = [
        'TranAD_QC',
        'TranAD_QC_learnable_cosine_complex_aware_complex_aware',
        'TranAD_QC_relu_mse_mse',
        'TranAD_QC_fixed_cosine_complex_aware_mse',
        'TranAD_C',
        'TranAD_C_some_suffix',
        'TranAD_Basic',
        'LSTM_AD'
    ]
    
    print("\n测试模型名称包含检查:")
    for name in test_names:
        # 测试新的检查逻辑
        is_tranad_c_or_qc = ('TranAD_C' in name or 'TranAD_QC' in name)
        print(f"  '{name}' -> {'TranAD_C/QC' if is_tranad_c_or_qc else 'Other'}")
    
    print("\n测试具体的模型实例:")
    
    # 创建不同配置的TranAD_QC模型
    configs = [
        ('learnable_cosine', 'complex_aware', 'complex_aware'),
        ('relu', 'mse', 'mse'),
        ('fixed_cosine', 'complex_aware', 'mse'),
    ]
    
    for act_type, loss_type, focus_type in configs:
        model = TranAD_QC(8, act_type, loss_type, focus_type)
        print(f"  配置: {act_type}_{loss_type}_{focus_type}")
        print(f"  模型名称: {model.name}")
        
        # 测试各种检查
        is_tranad_qc = 'TranAD_QC' in model.name
        is_tranad_c_or_qc = ('TranAD_C' in model.name or 'TranAD_QC' in model.name)
        
        print(f"  包含TranAD_QC: {is_tranad_qc}")
        print(f"  包含TranAD_C或TranAD_QC: {is_tranad_c_or_qc}")
        print()

def test_ablation_parameters():
    """测试消融研究参数是否正确传递"""
    print("=== 测试消融研究参数 ===")
    
    # 测试参数传递
    model = TranAD_QC(8, 'relu', 'mse', 'mse')
    
    print(f"模型名称: {model.name}")
    print(f"激活函数类型: {model.activation_type}")
    print(f"损失函数类型: {model.loss_type}")
    print(f"焦点分数类型: {model.focus_type}")
    
    # 验证名称包含所有参数
    expected_suffix = "_relu_mse_mse"
    if expected_suffix in model.name:
        print(f"✓ 模型名称正确包含参数后缀: {expected_suffix}")
    else:
        print(f"✗ 模型名称缺少参数后缀: {expected_suffix}")

if __name__ == '__main__':
    test_model_name_matching()
    test_ablation_parameters()
    
    print("\n" + "="*50)
    print("✓ 模型名称修复测试完成！")
    print("="*50)


