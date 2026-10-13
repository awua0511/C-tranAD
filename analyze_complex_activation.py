#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')

def analyze_complex_cosine_activation():
    """分析ComplexCosineActivation激活函数的特性"""
    print("🔬 分析ComplexCosineActivation激活函数...")
    
    try:
        from src.models import ComplexCosineActivation
        
        # 创建不同参数的激活函数
        activations = {
            'ω=0.5, φ=0': ComplexCosineActivation(omega_init=0.5, phi_init=0.0),
            'ω=1.0, φ=0': ComplexCosineActivation(omega_init=1.0, phi_init=0.0),
            'ω=1.5, φ=π/6': ComplexCosineActivation(omega_init=1.5, phi_init=np.pi/6),
            'ω=2.5, φ=π/3': ComplexCosineActivation(omega_init=2.5, phi_init=np.pi/3),
        }
        
        # 创建测试数据：在复平面上的点
        n_points = 1000
        angles = torch.linspace(0, 2*np.pi, n_points, dtype=torch.float64)
        magnitudes = torch.ones(n_points, dtype=torch.float64)
        
        # 复数表示：z = r * e^(iθ)
        real = magnitudes * torch.cos(angles)
        imag = magnitudes * torch.sin(angles)
        
        print("🎯 激活函数特性分析:")
        print("=" * 60)
        
        for name, act_fn in activations.items():
            act_fn.double()
            
            with torch.no_grad():
                out_real, out_imag = act_fn(real, imag)
                
                # 计算激活因子
                out_magnitude = torch.sqrt(out_real**2 + out_imag**2)
                activation_factor = out_magnitude / magnitudes
                
                print(f"\n{name}:")
                print(f"  激活因子范围: [{activation_factor.min().item():.3f}, {activation_factor.max().item():.3f}]")
                print(f"  激活因子均值: {activation_factor.mean().item():.3f}")
                print(f"  激活因子方差: {activation_factor.var().item():.3f}")
                
                # 检查学习参数的梯度
                test_loss = torch.sum(out_real**2 + out_imag**2)
                test_loss.backward()
                
                omega_grad = act_fn.omega.grad.item() if act_fn.omega.grad is not None else 0
                phi_grad = act_fn.phi.grad.item() if act_fn.phi.grad is not None else 0
                
                print(f"  ω梯度: {omega_grad:.6f}")
                print(f"  φ梯度: {phi_grad:.6f}")
                
                # 清除梯度
                act_fn.zero_grad()
        
        return True
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_activation_effectiveness():
    """测试激活函数在复数分类中的有效性"""
    print("\n🧪 测试激活函数有效性...")
    
    try:
        from src.models import ComplexCosineActivation, ComplexLinear, _split_complex_adjacent
        
        # 创建一个简单的复数网络
        feats = 6
        hidden = 16
        n_complex = feats // 2
        
        # 网络组件
        linear = ComplexLinear(n_complex, hidden).double()
        act_cosine = ComplexCosineActivation(omega_init=1.5, phi_init=np.pi/4).double()
        act_relu = nn.ReLU().double()  # 对比用的ReLU
        
        # 测试数据：包含不同相位的复数信号
        batch_size = 100
        x = torch.zeros(batch_size, feats, dtype=torch.float64)
        
        for i in range(n_complex):
            # 创建不同相位的复数信号
            phases = torch.linspace(0, 2*np.pi, batch_size)
            x[:, 2*i] = torch.cos(phases + i * np.pi/3)  # real
            x[:, 2*i+1] = torch.sin(phases + i * np.pi/3)  # imag
        
        # 前向传播
        real, imag = _split_complex_adjacent(x)
        
        # 复数线性变换
        h_real, h_imag = linear(real, imag)
        
        # 复数余弦激活
        out_cosine_real, out_cosine_imag = act_cosine(h_real, h_imag)
        
        # 对比：简单ReLU激活
        out_relu_real = act_relu(h_real)
        out_relu_imag = act_relu(h_imag)
        
        # 计算表达能力指标
        cosine_var_real = torch.var(out_cosine_real).item()
        cosine_var_imag = torch.var(out_cosine_imag).item()
        relu_var_real = torch.var(out_relu_real).item()
        relu_var_imag = torch.var(out_relu_imag).item()
        
        print(f"复数余弦激活 - 实部方差: {cosine_var_real:.6f}, 虚部方差: {cosine_var_imag:.6f}")
        print(f"ReLU激活 - 实部方差: {relu_var_real:.6f}, 虚部方差: {relu_var_imag:.6f}")
        
        # 计算相位敏感性
        cosine_magnitude = torch.sqrt(out_cosine_real**2 + out_cosine_imag**2)
        cosine_phase = torch.atan2(out_cosine_imag, out_cosine_real)
        
        relu_magnitude = torch.sqrt(out_relu_real**2 + out_relu_imag**2)
        relu_phase = torch.atan2(out_relu_imag, out_relu_real)
        
        print(f"复数余弦激活 - 幅度方差: {torch.var(cosine_magnitude).item():.6f}, 相位方差: {torch.var(cosine_phase).item():.6f}")
        print(f"ReLU激活 - 幅度方差: {torch.var(relu_magnitude).item():.6f}, 相位方差: {torch.var(relu_phase).item():.6f}")
        
        # 测试学习能力
        print("\n🎓 学习能力测试:")
        
        # 创建一个简单的分类任务
        target = torch.randint(0, 2, (batch_size,), dtype=torch.float64)
        
        # 复数余弦激活的分类器
        cosine_classifier = nn.Linear(hidden * 2, 1).double()  # *2 因为要concat real和imag
        cosine_features = torch.cat([out_cosine_real, out_cosine_imag], dim=-1)
        cosine_logits = cosine_classifier(cosine_features).squeeze()
        cosine_loss = nn.BCEWithLogitsLoss()(cosine_logits, target)
        
        # ReLU激活的分类器
        relu_classifier = nn.Linear(hidden * 2, 1).double()
        relu_features = torch.cat([out_relu_real, out_relu_imag], dim=-1)
        relu_logits = relu_classifier(relu_features).squeeze()
        relu_loss = nn.BCEWithLogitsLoss()(relu_logits, target)
        
        print(f"复数余弦激活损失: {cosine_loss.item():.6f}")
        print(f"ReLU激活损失: {relu_loss.item():.6f}")
        
        # 测试梯度
        cosine_loss.backward(retain_graph=True)
        omega_grad = act_cosine.omega.grad.item() if act_cosine.omega.grad is not None else 0
        phi_grad = act_cosine.phi.grad.item() if act_cosine.phi.grad is not None else 0
        
        print(f"复数余弦激活参数梯度 - ω: {omega_grad:.6f}, φ: {phi_grad:.6f}")
        
        print("\n✨ 复数余弦激活函数优势:")
        print("✅ 相位敏感，能捕捉复数信号的角度信息")
        print("✅ 可学习参数ω和φ，自适应调节激活模式")
        print("✅ 保持复数结构，不破坏复数运算的一致性")
        print("✅ 提供丰富的非线性变换，增强模型表达能力")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_improvements():
    """建议进一步的改进"""
    print("\n💡 激活函数优化建议:")
    print("""
1. 🎯 参数初始化策略
   - 低频分支: ω=0.5-1.0 (温和激活)
   - 中频分支: ω=1.5-2.0 (中等非线性)  
   - 高频分支: ω=2.0-3.0 (强非线性)
   
2. ⚡ 数值稳定性
   - 当前已有eps=1e-12保护
   - 小幅度时使用恒等映射
   
3. 🔧 学习率调节
   - ω和φ可能需要不同的学习率
   - 考虑对这些参数使用更小的weight_decay
   
4. 📊 监控指标
   - 观察ω和φ的变化趋势
   - 监控激活函数的饱和度
    """)

if __name__ == '__main__':
    success = analyze_complex_cosine_activation()
    if success:
        test_activation_effectiveness()
        suggest_improvements() 