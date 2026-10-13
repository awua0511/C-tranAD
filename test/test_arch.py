#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os
sys.path.append('..')

def test_architecture():
    """测试修复后的Transformer架构"""
    print("🔧 测试修复后的Transformer架构...")
    
    # 创建一个简单的测试
    try:
        from src.models import ComplexTransformerEncoderLayer, ComplexTransformerDecoderLayer
        
        # 测试参数
        d_model = 64
        nhead = 4
        ff_dim = 256
        seq_len = 10
        batch_size = 2
        
        print(f"测试参数: d_model={d_model}, nhead={nhead}, ff_dim={ff_dim}")
        print(f"每个head维度: {d_model // nhead}")
        
        # 创建测试数据
        src_r = torch.randn(seq_len, batch_size, d_model, dtype=torch.float64)
        src_i = torch.randn(seq_len, batch_size, d_model, dtype=torch.float64)
        
        # 测试Encoder Layer
        print("\n📍 测试 ComplexTransformerEncoderLayer...")
        encoder_layer = ComplexTransformerEncoderLayer(d_model, nhead, ff_dim).double()
        out_r, out_i = encoder_layer(src_r, src_i)
        print(f"✅ Encoder输出形状: real={out_r.shape}, imag={out_i.shape}")
        
        # 测试Decoder Layer
        print("\n📍 测试 ComplexTransformerDecoderLayer...")
        decoder_layer = ComplexTransformerDecoderLayer(d_model, nhead, ff_dim).double()
        mem_r, mem_i = out_r, out_i  # 使用encoder输出作为memory
        tgt_r = torch.randn(1, batch_size, d_model, dtype=torch.float64)
        tgt_i = torch.randn(1, batch_size, d_model, dtype=torch.float64)
        
        dec_out_r, dec_out_i = decoder_layer(tgt_r, tgt_i, mem_r, mem_i)
        print(f"✅ Decoder输出形状: real={dec_out_r.shape}, imag={dec_out_i.shape}")
        
        # 检查数值稳定性
        def check_values(tensor, name):
            nan_count = torch.isnan(tensor).sum()
            inf_count = torch.isinf(tensor).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"⚠️  {name}: NaN={nan_count}, Inf={inf_count}")
            else:
                print(f"✅ {name}: 数值稳定")
                
        check_values(out_r, "Encoder real")
        check_values(out_i, "Encoder imag") 
        check_values(dec_out_r, "Decoder real")
        check_values(dec_out_i, "Decoder imag")
        
        print(f"\n🎉 架构测试通过！")
        print(f"📊 Encoder参数数量: {sum(p.numel() for p in encoder_layer.parameters())}")
        print(f"📊 Decoder参数数量: {sum(p.numel() for p in decoder_layer.parameters())}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_architecture() 