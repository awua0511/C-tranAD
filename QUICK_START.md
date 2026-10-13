# 快速开始指南

## 基于你的训练命令的完整示例

### 1. 环境准备

```bash
# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn tqdm scipy
```

### 2. 数据准备

确保你的数据在 `data/my_data/` 目录下，然后运行预处理：

```bash
python preprocess_2_mydata.py
```

### 3. 执行你的训练命令

```bash
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
```

### 4. 命令参数详解

| 参数 | 值 | 说明 |
|------|----|----- |
| `--model` | TranAD_QC | 使用TranAD_QC模型（支持消融研究） |
| `--dataset` | my_data | 使用my_data数据集 |
| `--train_clf` | - | 训练分类头 |
| `--retrain` | - | 重新训练（覆盖已有模型） |
| `--use_clf` | - | 使用分类头进行评估 |
| `--activation_type` | learnable_cosine | 使用可学习余弦激活函数 |
| `--loss_type` | complex_aware | 使用复数感知损失函数 |
| `--focus_type` | complex_aware | 使用复数感知焦点分数 |
| `--clf_version` | v2 | 使用v2版本的分类器 |
| `--num_epochs` | 10 | 训练10个epoch |
| `--visualize` | - | 生成可视化图表 |
| `--experiment_name` | CV-FCNN_learnable_cosine | 实验名称 |

### 5. 预期输出

训练完成后，你会得到：

#### 5.1 模型检查点
```
checkpoints/CV-FCNN_learnable_cosine/model.ckpt
```

#### 5.2 可视化结果
```
plots/classification/CV-FCNN_learnable_cosine/
├── classification_boundary_pca.png      # PCA分类边界
├── classification_boundary_tsne.png     # t-SNE分类边界
├── confusion_matrix.png                 # 混淆矩阵
├── roc_curve.png                        # ROC曲线
├── f1_score_analysis.png                # F1-score分析（新增）
├── feature_importance.png               # 特征重要性
├── multi_fold_analysis.png              # 5-fold分析
└── classification_report.txt            # 详细报告
```

#### 5.3 控制台输出
```
=== 开始5折交叉验证可视化 ===
处理第 1 折...
处理第 2 折...
处理第 3 折...
处理第 4 折...
处理第 5 折...

✓ 完整可视化报告已生成到: plots/classification/CV-FCNN_learnable_cosine

Main Performance Metrics:
Accuracy: 0.xxxx
F1-Score: 0.xxxx
AUC Score: 0.xxxx
```

### 6. 测试已训练模型

```bash
python main.py \
    --model TranAD_QC \
    --dataset my_data \
    --test \
    --use_clf \
    --experiment_name CV-FCNN_learnable_cosine
```

### 7. 消融研究示例

基于你的配置，可以尝试其他激活函数：

```bash
# 使用ReLU激活函数
python main.py \
    --model TranAD_QC \
    --dataset my_data \
    --train_clf \
    --retrain \
    --use_clf \
    --activation_type relu \
    --loss_type complex_aware \
    --focus_type complex_aware \
    --clf_version v2 \
    --num_epochs 10 \
    --visualize \
    --experiment_name CV-FCNN_relu

# 使用LeakyReLU激活函数
python main.py \
    --model TranAD_QC \
    --dataset my_data \
    --train_clf \
    --retrain \
    --use_clf \
    --activation_type leaky_relu \
    --loss_type complex_aware \
    --focus_type complex_aware \
    --clf_version v2 \
    --num_epochs 10 \
    --visualize \
    --experiment_name CV-FCNN_leaky_relu
```

### 8. 故障排除

#### 8.1 如果遇到内存不足
```bash
# 减少训练轮数
--num_epochs 5
```

#### 8.2 如果可视化失败
```bash
# 检查matplotlib后端
export MPLBACKEND=Agg
```

#### 8.3 如果模型加载失败
```bash
# 确保checkpoint路径正确
ls checkpoints/CV-FCNN_learnable_cosine/
```

### 9. 结果分析

#### 9.1 查看F1-score分析
打开 `f1_score_analysis.png` 查看：
- 不同阈值下的F1-score变化
- 最佳F1-score和对应阈值
- 5-fold交叉验证的F1-score分布
- 性能指标总结

#### 9.2 查看分类边界
打开 `classification_boundary_pca.png` 和 `classification_boundary_tsne.png` 查看：
- 数据在2D空间的分布
- 分类边界
- 正常/异常样本的分离情况

#### 9.3 查看详细报告
打开 `classification_report.txt` 查看：
- 总体性能指标
- 每个fold的性能
- 数据分布统计

### 10. 下一步

1. **比较不同配置**: 运行多个实验并比较结果
2. **调整超参数**: 尝试不同的学习率、批次大小等
3. **数据增强**: 尝试不同的数据预处理方法
4. **模型集成**: 结合多个模型的结果

### 11. 常用命令组合

```bash
# 快速测试（不训练）
python main.py --model TranAD_QC --dataset my_data --test --use_clf --experiment_name CV-FCNN_learnable_cosine

# 只生成可视化（使用已有模型）
python main.py --model TranAD_QC --dataset my_data --test --use_clf --visualize --experiment_name CV-FCNN_learnable_cosine

# 训练更多轮数
python main.py --model TranAD_QC --dataset my_data --train_clf --retrain --use_clf --num_epochs 20 --experiment_name CV-FCNN_learnable_cosine_20epochs
```

这样你就可以轻松地开始使用这个项目了！
