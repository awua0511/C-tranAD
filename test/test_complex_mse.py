#!/usr/bin/env python3
"""
测试ComplexMSELoss的正确性
"""

import torch
import torch.nn as nn
import numpy as np

# 导入我们的ComplexMSELoss
import sys
sys.path.append('../src')
from models import ComplexMSELoss

def test_complex_mse():
    """测试复数MSE损失函数的正确性"""
    print("🧪 测试ComplexMSELoss...")
    
    # 创建ComplexMSELoss实例
    complex_mse = ComplexMSELoss()
    
    # 测试1: 简单的复数MSE
    print("\n📊 测试1: 简单复数MSE")
    
    # 创建测试数据：使用_merge_complex_adjacent格式 (r0,i0,r1,i1,...)
    pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)  # 复数: 1+2j, 3+4j
    target = torch.tensor([[[0.5, 1.5, 2.5, 3.5]]], dtype=torch.float64)  # 复数: 0.5+1.5j, 2.5+3.5j
    
    print(f"预测值: {pred}")
    print(f"目标值: {target}")
    
    # 计算损失
    loss = complex_mse(pred, target)
    print(f"计算的损失: {loss}")
    
    # 手动验证
    # 第一个复数: (1+2j) - (0.5+1.5j) = 0.5+0.5j
    # |0.5+0.5j|² = 0.5² + 0.5² = 0.25 + 0.25 = 0.5
    # 第二个复数: (3+4j) - (2.5+3.5j) = 0.5+0.5j
    # |0.5+0.5j|² = 0.5² + 0.5² = 0.25 + 0.25 = 0.5
    expected_loss = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
    print(f"期望的损失: {expected_loss}")
    
    # 验证结果
    is_correct = torch.allclose(loss, expected_loss, atol=1e-6)
    print(f"✅ 测试1 {'通过' if is_correct else '失败'}")
    
    # 测试2: 批量数据
    print("\n📊 测试2: 批量数据")
    
    batch_pred = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0]],  # 第一个样本
        [[2.0, 3.0, 4.0, 5.0]]   # 第二个样本
    ], dtype=torch.float64)
    
    batch_target = torch.tensor([
        [[0.5, 1.5, 2.5, 3.5]],  # 第一个样本
        [[1.5, 2.5, 3.5, 4.5]]   # 第二个样本
    ], dtype=torch.float64)
    
    batch_loss = complex_mse(batch_pred, batch_target)
    print(f"批量损失形状: {batch_loss.shape}")
    print(f"批量损失: {batch_loss}")
    
    # 测试3: 与标准MSE比较
    print("\n📊 测试3: 与标准MSE比较")
    
    # 标准MSE应该等于复数MSE（因为复数MSE就是实部和虚部MSE的和）
    standard_mse = nn.MSELoss(reduction='none')
    standard_loss = standard_mse(pred, target)
    print(f"标准MSE损失: {standard_loss}")
    
    # 复数MSE应该等于标准MSE
    complex_loss = complex_mse(pred, target)
    print(f"复数MSE损失: {complex_loss}")
    
    # 验证复数MSE等于标准MSE
    is_equal = torch.allclose(complex_loss, standard_loss, atol=1e-6)
    print(f"✅ 复数MSE与标准MSE {'相等' if is_equal else '不相等'}")
    
    # 测试4: 零损失情况
    print("\n📊 测试4: 零损失情况")
    
    same_pred = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)
    same_target = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float64)
    
    zero_loss = complex_mse(same_pred, same_target)
    print(f"相同预测和目标的损失: {zero_loss}")
    
    is_zero = torch.allclose(zero_loss, torch.zeros_like(zero_loss), atol=1e-6)
    print(f"✅ 零损失测试 {'通过' if is_zero else '失败'}")
    
    print("\n🎉 所有测试完成！")

if __name__ == "__main__":
    test_complex_mse()


