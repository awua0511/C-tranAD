#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')

def test_classifier_heads():
    """测试不同版本的复数分类头"""
    print("🧪 测试复数分类头版本...")
    
    try:
        from src.models import ComplexClfHead, ComplexClfHeadV2, ComplexClfHeadV3
        
        # 测试参数
        feats = 10  # 必须是偶数
        batch_size = 32
        
        # 创建测试数据
        x = torch.randn(batch_size, feats, dtype=torch.float64)
        
        print(f"输入特征维度: {feats}, 批次大小: {batch_size}")
        print("-" * 60)
        
        # 测试三个版本
        versions = {
            'V1 (原始)': ComplexClfHead(feats),
            'V2 (多尺度)': ComplexClfHeadV2(feats),
            'V3 (注意力)': ComplexClfHeadV3(feats)
        }
        
        for name, clf in versions.items():
            clf = clf.double()
            
            # 前向传播
            with torch.no_grad():
                logits = clf(x)
            
            # 统计信息
            param_count = sum(p.numel() for p in clf.parameters())
            output_range = [logits.min().item(), logits.max().item()]
            output_mean = logits.mean().item()
            
            print(f"{name}:")
            print(f"  参数数量: {param_count:,}")
            print(f"  输出形状: {logits.shape}")
            print(f"  输出范围: [{output_range[0]:.4f}, {output_range[1]:.4f}]")
            print(f"  输出均值: {output_mean:.4f}")
            print()
        
        print("✅ 所有分类头测试通过!")
        
        # 给出建议
        print("\n💡 使用建议:")
        print("- V1: 轻量级，参数少，适合快速测试")
        print("- V2: 多尺度特征，更好的特征表达，适合中等规模数据")
        print("- V3: 注意力机制，最强表达能力，适合复杂异常模式")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_usage():
    """使用建议"""
    print("\n🎯 如何使用不同版本:")
    print("""
# 使用V2版本（多尺度特征）
python main.py --model TranAD_QC --dataset my_data --train_clf --clf_version v2

# 使用V3版本（注意力机制）  
python main.py --model TranAD_QC --dataset my_data --train_clf --clf_version v3

# 使用默认V1版本
python main.py --model TranAD_QC --dataset my_data --train_clf
    """)

if __name__ == '__main__':
    if test_classifier_heads():
        suggest_usage() 