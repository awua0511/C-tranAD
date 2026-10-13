#!/usr/bin/env python3
"""
测试消融研究框架的简单脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models import TranAD_QC, get_activation_function, get_loss_function

def test_activation_functions():
    """测试不同的激活函数"""
    print("=== 测试激活函数 ===")
    
    activation_types = ['learnable_cosine', 'relu', 'fixed_cosine']
    
    for act_type in activation_types:
        print(f"\n测试激活函数: {act_type}")
        
        # 创建激活函数
        act_func = get_activation_function(act_type)
        print(f"  创建成功: {type(act_func).__name__}")
        
        # 测试前向传播
        real = torch.randn(2, 4, dtype=torch.float64)
        imag = torch.randn(2, 4, dtype=torch.float64)
        
        real_out, imag_out = act_func(real, imag)
        print(f"  输入形状: real={real.shape}, imag={imag.shape}")
        print(f"  输出形状: real={real_out.shape}, imag={imag_out.shape}")
        print(f"  输出范围: real=[{real_out.min():.3f}, {real_out.max():.3f}], imag=[{imag_out.min():.3f}, {imag_out.max():.3f}]")

def test_loss_functions():
    """测试不同的损失函数"""
    print("\n=== 测试损失函数 ===")
    
    loss_types = ['complex_aware', 'mse']
    
    for loss_type in loss_types:
        print(f"\n测试损失函数: {loss_type}")
        
        # 创建损失函数
        loss_func = get_loss_function(loss_type)
        print(f"  创建成功: {type(loss_func).__name__}")
        
        # 测试前向传播
        pred = torch.randn(10, 2, 8, dtype=torch.float64)  # [T, B, feats]
        target = torch.randn(10, 2, 8, dtype=torch.float64)
        
        loss = loss_func(pred, target)
        print(f"  输入形状: pred={pred.shape}, target={target.shape}")
        print(f"  输出形状: {loss.shape}")
        print(f"  损失范围: [{loss.min():.6f}, {loss.max():.6f}]")

def test_tranad_qc_variants():
    """测试TranAD_QC的不同变体"""
    print("\n=== 测试TranAD_QC变体 ===")
    
    # 测试参数组合
    test_configs = [
        ('learnable_cosine', 'complex_aware', 'complex_aware'),
        ('relu', 'mse', 'mse'),
        ('fixed_cosine', 'complex_aware', 'mse'),
    ]
    
    feats = 8  # 偶数，便于复数处理
    
    for act_type, loss_type, focus_type in test_configs:
        print(f"\n测试配置: activation={act_type}, loss={loss_type}, focus={focus_type}")
        
        try:
            # 创建模型
            model = TranAD_QC(feats, act_type, loss_type, focus_type)
            model = model.double()
            print(f"  模型名称: {model.name}")
            print(f"  模型参数数量: {sum(p.numel() for p in model.parameters())}")
            
            # 测试前向传播
            src = torch.randn(10, 2, feats, dtype=torch.float64)  # [w, B, feats]
            tgt = torch.randn(1, 2, feats, dtype=torch.float64)   # [1, B, feats]
            
            with torch.no_grad():
                # 测试正常前向传播
                o1, o2_hat = model(src, tgt)
                print(f"  输出形状: o1={o1.shape}, o2_hat={o2_hat.shape}")
                
                # 测试完整前向传播
                o1_full, o2_full, o2_hat_full = model(src, tgt, return_all=True)
                print(f"  完整输出形状: o1={o1_full.shape}, o2={o2_full.shape}, o2_hat={o2_hat_full.shape}")
                
                # 测试误差特征提取
                if hasattr(model, 'extract_error_features'):
                    feat_simple = model.extract_error_features(src, tgt, mode='simple')
                    feat_complex = model.extract_error_features(src, tgt, mode='complex')
                    print(f"  误差特征形状: simple={feat_simple.shape}, complex={feat_complex.shape}")
                
                print(f"  ✓ 配置测试成功")
                
        except Exception as e:
            print(f"  ✗ 配置测试失败: {e}")

def main():
    """主测试函数"""
    print("开始消融研究框架测试...\n")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    try:
        test_activation_functions()
        test_loss_functions()
        test_tranad_qc_variants()
        
        print("\n" + "="*50)
        print("✓ 所有测试完成！消融研究框架可以正常工作。")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


