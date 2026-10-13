#!/usr/bin/env python3

import torch
import sys
import os
sys.path.append('..')

from src.models import TranAD_QC

def test_tranad_qc():
    print('测试TranAD_QC模型基本功能...')
    
    feats = 10  # 偶数特征
    try:
        model = TranAD_QC(feats).double()
        print(f'模型创建成功: {model.name}')
        
        # 创建测试数据
        batch_size = 4
        window_size = 10
        src = torch.randn(window_size, batch_size, feats, dtype=torch.float64)
        tgt = torch.randn(1, batch_size, feats, dtype=torch.float64)
        
        print(f'输入形状: src={src.shape}, tgt={tgt.shape}')
        
        # 测试前向传播
        o1, o2, o2_hat = model(src, tgt, return_all=True)
        print(f'前向传播成功: o1={o1.shape}, o2={o2.shape}, o2_hat={o2_hat.shape}')
        
        # 检查数值范围
        print(f'o1 范围: [{o1.min().item():.4f}, {o1.max().item():.4f}]')
        print(f'o2_hat 范围: [{o2_hat.min().item():.4f}, {o2_hat.max().item():.4f}]')
        
        # 检查NaN和Inf
        nan_count = torch.isnan(o1).sum() + torch.isnan(o2_hat).sum()
        inf_count = torch.isinf(o1).sum() + torch.isinf(o2_hat).sum()
        
        if nan_count > 0:
            print(f'警告: 发现 {nan_count} 个NaN值')
        if inf_count > 0:
            print(f'警告: 发现 {inf_count} 个Inf值')
        
        if nan_count == 0 and inf_count == 0:
            print('✓ 输出数值正常，无NaN或Inf')
        
        # 测试误差特征提取
        error_feat = model.extract_error_features(src, tgt, mode='simple')
        print(f'误差特征提取成功: {error_feat.shape}')
        print(f'误差特征范围: [{error_feat.min().item():.4f}, {error_feat.max().item():.4f}]')
        
        # 检查误差特征的NaN和Inf
        feat_nan = torch.isnan(error_feat).sum()
        feat_inf = torch.isinf(error_feat).sum()
        
        if feat_nan > 0:
            print(f'警告: 误差特征中发现 {feat_nan} 个NaN值')
        if feat_inf > 0:
            print(f'警告: 误差特征中发现 {feat_inf} 个Inf值')
        
        if feat_nan == 0 and feat_inf == 0:
            print('✓ 误差特征数值正常')
        
        # 测试分类器
        if hasattr(model, 'classifier'):
            logits = model.predict_scores(src, tgt)
            print(f'分类器输出: {logits.shape}, 范围: [{logits.min().item():.4f}, {logits.max().item():.4f}]')
            
            clf_nan = torch.isnan(logits).sum()
            clf_inf = torch.isinf(logits).sum()
            
            if clf_nan == 0 and clf_inf == 0:
                print('✓ 分类器输出数值正常')
            else:
                print(f'警告: 分类器输出异常 - NaN: {clf_nan}, Inf: {clf_inf}')
        
        print('\n🎉 TranAD_QC模型所有测试通过！')
        return True
        
    except Exception as e:
        print(f'❌ 错误: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    test_tranad_qc() 