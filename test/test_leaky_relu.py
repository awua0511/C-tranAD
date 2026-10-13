#!/usr/bin/env python3
"""
简单测试LeakyReLU激活函数
"""

import torch
import sys
sys.path.append('../src')

def test_leaky_relu():
    print("🧪 测试ComplexLeakyReLU...")
    
    try:
        from models import ComplexLeakyReLU
        
        # 创建LeakyReLU实例
        leaky_relu = ComplexLeakyReLU(negative_slope=0.01)
        
        # 测试数据
        real = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=torch.float64)
        imag = torch.tensor([[-1.0, 0.0, 1.0, -0.5, 0.5]], dtype=torch.float64)
        
        print(f"输入实部: {real}")
        print(f"输入虚部: {imag}")
        
        # 前向传播
        out_real, out_imag = leaky_relu(real, imag)
        
        print(f"输出实部: {out_real}")
        print(f"输出虚部: {out_imag}")
        
        # 验证结果
        expected_real = torch.tensor([[-0.02, -0.01, 0.0, 1.0, 2.0]], dtype=torch.float64)
        expected_imag = torch.tensor([[-0.01, 0.0, 1.0, -0.005, 0.5]], dtype=torch.float64)
        
        is_correct = (torch.allclose(out_real, expected_real, atol=1e-6) and 
                     torch.allclose(out_imag, expected_imag, atol=1e-6))
        
        print(f"✅ LeakyReLU测试 {'通过' if is_correct else '失败'}")
        
        # 测试工厂函数
        from models import get_activation_function
        leaky_from_factory = get_activation_function('leaky_relu')
        factory_real, factory_imag = leaky_from_factory(real, imag)
        
        is_same = (torch.allclose(out_real, factory_real) and 
                  torch.allclose(out_imag, factory_imag))
        
        print(f"✅ 工厂函数测试 {'通过' if is_same else '失败'}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_leaky_relu()


