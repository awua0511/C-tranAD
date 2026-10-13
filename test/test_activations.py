#!/usr/bin/env python3
"""
测试所有复数激活函数
"""

import torch
import torch.nn as nn
import numpy as np

# 导入我们的激活函数
import sys
sys.path.append('../src')
from models import (ComplexReLU, ComplexLeakyReLU, ComplexCosineActivation, 
                   ComplexFixedCosineActivation, get_activation_function)

def test_activations():
    """测试所有复数激活函数"""
    print("🧪 测试复数激活函数...")
    
    # 创建测试数据
    real = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    imag = torch.tensor([[-1.0, 0.0, 1.0, -0.5, 0.5]], dtype=torch.float64)
    
    print(f"输入实部: {real}")
    print(f"输入虚部: {imag}")
    
    # 测试1: ComplexReLU
    print("\n📊 测试1: ComplexReLU")
    relu = ComplexReLU()
    relu_real, relu_imag = relu(real, imag)
    print(f"ReLU输出实部: {relu_real}")
    print(f"ReLU输出虚部: {relu_imag}")
    
    # 验证ReLU行为：负数变为0，正数保持不变
    expected_relu_real = torch.tensor([[0.0, 0.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
    expected_relu_imag = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.5]], dtype=torch.float64)
    
    is_correct = (torch.allclose(relu_real, expected_relu_real) and 
                 torch.allclose(relu_imag, expected_relu_imag))
    print(f"✅ ComplexReLU测试 {'通过' if is_correct else '失败'}")
    
    # 测试2: ComplexLeakyReLU
    print("\n📊 测试2: ComplexLeakyReLU")
    leaky_relu = ComplexLeakyReLU(negative_slope=0.01)
    leaky_real, leaky_imag = leaky_relu(real, imag)
    print(f"LeakyReLU输出实部: {leaky_real}")
    print(f"LeakyReLU输出虚部: {leaky_imag}")
    
    # 验证LeakyReLU行为：负数乘以0.01，正数保持不变
    expected_leaky_real = torch.tensor([[-0.02, -0.01, 0.0, 1.0, 2.0]], dtype=torch.float64)
    expected_leaky_imag = torch.tensor([[-0.01, 0.0, 1.0, -0.005, 0.5]], dtype=torch.float64)
    
    is_correct = (torch.allclose(leaky_real, expected_leaky_real, atol=1e-6) and 
                 torch.allclose(leaky_imag, expected_leaky_imag, atol=1e-6))
    print(f"✅ ComplexLeakyReLU测试 {'通过' if is_correct else '失败'}")
    
    # 测试3: ComplexCosineActivation
    print("\n📊 测试3: ComplexCosineActivation")
    cos_act = ComplexCosineActivation(omega_init=1.0, phi_init=0.0)
    cos_real, cos_imag = cos_act(real, imag)
    print(f"Cosine激活输出实部: {cos_real}")
    print(f"Cosine激活输出虚部: {cos_imag}")
    
    # 测试4: ComplexFixedCosineActivation
    print("\n📊 测试4: ComplexFixedCosineActivation")
    fixed_cos = ComplexFixedCosineActivation(omega=1.0, phi=0.0)
    fixed_real, fixed_imag = fixed_cos(real, imag)
    print(f"固定Cosine激活输出实部: {fixed_real}")
    print(f"固定Cosine激活输出虚部: {fixed_imag}")
    
    # 测试5: 工厂函数
    print("\n📊 测试5: 激活函数工厂")
    
    # 测试ReLU
    relu_from_factory = get_activation_function('relu')
    factory_real, factory_imag = relu_from_factory(real, imag)
    is_same = (torch.allclose(relu_real, factory_real) and 
              torch.allclose(relu_imag, factory_imag))
    print(f"✅ 工厂函数ReLU {'正确' if is_same else '错误'}")
    
    # 测试LeakyReLU
    leaky_from_factory = get_activation_function('leaky_relu')
    factory_leaky_real, factory_leaky_imag = leaky_from_factory(real, imag)
    is_same = (torch.allclose(leaky_real, factory_leaky_real) and 
              torch.allclose(leaky_imag, factory_leaky_imag))
    print(f"✅ 工厂函数LeakyReLU {'正确' if is_same else '错误'}")
    
    # 测试默认激活函数
    default_act = get_activation_function('learnable_cosine')
    default_real, default_imag = default_act(real, imag)
    print(f"✅ 默认激活函数 (learnable_cosine) 创建成功")
    
    # 测试错误输入
    try:
        wrong_act = get_activation_function('invalid_type')
        print("❌ 应该抛出错误但没有")
    except ValueError as e:
        print(f"✅ 正确捕获错误: {e}")
    
    print("\n🎉 所有激活函数测试完成！")

def test_activation_properties():
    """测试激活函数的数学性质"""
    print("\n🔬 测试激活函数的数学性质...")
    
    # 测试零输入
    zero_real = torch.zeros(1, 5, dtype=torch.float64)
    zero_imag = torch.zeros(1, 5, dtype=torch.float64)
    
    # ReLU在零点的行为
    relu = ComplexReLU()
    relu_zero_real, relu_zero_imag = relu(zero_real, zero_imag)
    print(f"ReLU在零点: 实部={relu_zero_real}, 虚部={relu_zero_imag}")
    
    # LeakyReLU在零点的行为
    leaky_relu = ComplexLeakyReLU()
    leaky_zero_real, leaky_zero_imag = leaky_relu(zero_real, zero_imag)
    print(f"LeakyReLU在零点: 实部={leaky_zero_real}, 虚部={leaky_zero_imag}")
    
    # 测试连续性
    small_positive = torch.tensor([[0.001]], dtype=torch.float64)
    small_negative = torch.tensor([[-0.001]], dtype=torch.float64)
    
    relu_pos = relu(small_positive, zero_real[:, :1])
    relu_neg = relu(small_negative, zero_real[:, :1])
    print(f"ReLU在正小值: {relu_pos}")
    print(f"ReLU在负小值: {relu_neg}")
    
    leaky_pos = leaky_relu(small_positive, zero_real[:, :1])
    leaky_neg = leaky_relu(small_negative, zero_real[:, :1])
    print(f"LeakyReLU在正小值: {leaky_pos}")
    print(f"LeakyReLU在负小值: {leaky_neg}")

if __name__ == "__main__":
    test_activations()
    test_activation_properties()


