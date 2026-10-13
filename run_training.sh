#!/bin/bash

# TranAD_QC 训练脚本
# 基于你的训练参数

echo "开始训练 TranAD_QC 模型..."
echo "实验名称: CV-FCNN_learnable_cosine"
echo "=================================="

python main.py \
    --model TranAD_QC \
    --dataset my_data \
    --train_clf \
    --retrain \
    --use_clf \
    --activation_type learnable_cosine \
    --loss_type complex_aware \
    --focus_type complex_aware \
    --clf_version v2 \
    --num_epochs 10 \
    --visualize \
    --experiment_name CV-FCNN_learnable_cosine

echo "=================================="
echo "训练完成！"
echo "结果保存在:"
echo "  - 模型检查点: checkpoints/CV-FCNN_learnable_cosine/model.ckpt"
echo "  - 可视化结果: plots/classification/CV-FCNN_learnable_cosine/"
echo "=================================="
