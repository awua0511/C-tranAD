#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path

def clean_checkpoints(model_name=None, dataset_name=None, confirm=True):
    """清理checkpoints目录"""
    
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        print("❌ checkpoints目录不存在")
        return
    
    # 查找要清理的目录
    if model_name and dataset_name:
        target_dirs = [checkpoint_dir / f"{model_name}_{dataset_name}"]
    elif model_name:
        target_dirs = list(checkpoint_dir.glob(f"{model_name}_*"))
    elif dataset_name:
        target_dirs = list(checkpoint_dir.glob(f"*_{dataset_name}"))
    else:
        target_dirs = list(checkpoint_dir.glob("*"))
    
    if not target_dirs:
        print("✅ 没有找到要清理的checkpoints")
        return
    
    print("🗂️  找到以下checkpoints:")
    total_size = 0
    for target_dir in target_dirs:
        if target_dir.is_dir():
            size = sum(f.stat().st_size for f in target_dir.rglob('*') if f.is_file())
            total_size += size
            print(f"  📁 {target_dir.name} ({size/1024/1024:.1f} MB)")
    
    print(f"\n📊 总大小: {total_size/1024/1024:.1f} MB")
    
    if confirm:
        response = input(f"\n❓ 确定要删除这些checkpoints吗? [y/N]: ")
        if response.lower() != 'y':
            print("🚫 取消清理")
            return
    
    # 执行清理
    deleted_count = 0
    for target_dir in target_dirs:
        if target_dir.is_dir():
            try:
                shutil.rmtree(target_dir)
                print(f"✅ 已删除: {target_dir.name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ 删除失败 {target_dir.name}: {e}")
    
    print(f"\n🎉 清理完成! 删除了 {deleted_count} 个checkpoint目录")

def main():
    parser = argparse.ArgumentParser(description='清理TranAD checkpoints')
    parser.add_argument('--model', type=str, help='模型名称 (如: TranAD_QC)')
    parser.add_argument('--dataset', type=str, help='数据集名称 (如: my_data)')
    parser.add_argument('--force', action='store_true', help='强制删除，不询问确认')
    
    args = parser.parse_args()
    
    print("🧹 TranAD Checkpoint 清理工具")
    print("=" * 40)
    
    clean_checkpoints(
        model_name=args.model,
        dataset_name=args.dataset, 
        confirm=not args.force
    )

if __name__ == '__main__':
    main() 