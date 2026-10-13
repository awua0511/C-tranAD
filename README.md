# TranAD 时间序列异常检测项目

## 项目简介

本项目是基于TranAD（Transformer-based Anomaly Detection）的时间序列异常检测系统，支持多种模型变体和消融研究。项目包含完整的训练、测试和可视化功能，特别针对TranAD_QC模型进行了优化。

## 主要特性

- **多种模型支持**: TranAD、TranAD_C、TranAD_QC等
- **消融研究**: 支持不同激活函数、损失函数的组合实验
- **分类头训练**: 支持在TranAD基础上训练分类器
- **完整可视化**: 包含F1-score分析、混淆矩阵、ROC曲线等
- **5-fold交叉验证**: 提供可靠的模型评估
- **实验管理**: 支持自定义实验名称和结果管理

## 环境要求

```bash
# Python 3.7+
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
pip install scipy
```

## 快速开始

### 1. 基本训练

```bash
# 训练TranAD_QC模型 NAB
python main.py --model TranAD_QC --dataset my_data --train_clf --use_clf --visualize
```

### 2. 完整训练命令（基于你的参数）

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

### 3. 参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `--model` | 模型类型 | TranAD, TranAD_C, TranAD_QC |
| `--dataset` | 数据集 | my_data, MBA, SMD, SWaT等 |
| `--train_clf` | 训练分类头 | 布尔值 |
| `--use_clf` | 使用分类头评估 | 布尔值 |
| `--retrain` | 重新训练模型 | 布尔值 |
| `--activation_type` | 激活函数类型 | learnable_cosine, relu, leaky_relu, fixed_cosine |
| `--loss_type` | 损失函数类型 | complex_aware, mse |
| `--focus_type` | 焦点分数计算 | complex_aware, mse |
| `--clf_version` | 分类器版本 | v1, v2, v3 |
| `--num_epochs` | 训练轮数 | 整数 |
| `--visualize` | 生成可视化 | 布尔值 |
| `--experiment_name` | 实验名称 | 自定义字符串 |

## 数据集准备

### 1. 数据格式

数据应放在 `data/` 目录下，支持以下格式：
- CSV文件
- NumPy数组文件 (.npy)
- Excel文件 (.xlsx)

### 2. 数据预处理

```bash
# 预处理my_data数据集
python preprocess_2_mydata.py
```

### 3. 支持的数据集

- `my_data`: 自定义数据集
- `MBA`: 多变量时间序列数据集
- `SMD`: 服务器机器数据集
- `SWaT`: 水处理系统数据集
- `synthetic`: 合成数据集

## 模型训练

### 1. 基础训练

```bash
# 训练基础TranAD模型
python main.py --model TranAD --dataset my_data
```

### 2. 带分类头的训练

```bash
# 训练TranAD_C模型（带分类头）
python main.py --model TranAD_C --dataset my_data --train_clf --use_clf
```

### 3. 消融研究训练

```bash
# 不同激活函数的实验
python main.py --model TranAD_QC --dataset my_data --activation_type relu --experiment_name relu_experiment --visualize

python main.py --model TranAD_QC --dataset my_data --activation_type leaky_relu --experiment_name leaky_relu_experiment --visualize

# 不同损失函数的实验
python main.py --model TranAD_QC --dataset my_data --loss_type mse --experiment_name mse_experiment --visualize
```

## 模型测试

### 1. 测试已训练模型

```bash
# 测试模型（不重新训练）
python main.py --model TranAD_QC --dataset my_data --test --experiment_name CV-FCNN_learnable_cosine
```

### 2. 使用分类头测试

```bash
# 使用分类头进行评估
python main.py --model TranAD_QC --dataset my_data --test --use_clf --experiment_name CV-FCNN_learnable_cosine
```

## 可视化功能

### 1. 生成完整可视化报告

```bash
python main.py --model TranAD_QC --dataset my_data --visualize --experiment_name my_experiment
```

### 2. 可视化输出

可视化结果保存在 `plots/classification/{experiment_name}/` 目录下：

- `classification_boundary_pca.png`: PCA降维分类边界
- `classification_boundary_tsne.png`: t-SNE降维分类边界
- `confusion_matrix.png`: 混淆矩阵
- `roc_curve.png`: ROC曲线
- `f1_score_analysis.png`: F1-score分析
- `feature_importance.png`: 特征重要性
- `multi_fold_analysis.png`: 多fold分析
- `classification_report.txt`: 详细统计报告

### 3. F1-score分析

新增的F1-score可视化包括：
- 不同阈值下的F1-score变化
- Precision、Recall、F1-score对比
- 5-fold交叉验证的F1-score分布
- 性能指标总结

## 实验管理

### 1. 实验命名

```bash
# 使用自定义实验名称
python main.py --model TranAD_QC --dataset my_data --experiment_name "baseline_v1" --visualize

# 自动生成实验名称（基于参数组合）
python main.py --model TranAD_QC --dataset my_data --activation_type relu --loss_type mse --visualize
```

### 2. 批量实验

创建批量实验脚本：

```bash
#!/bin/bash

# 定义实验配置
experiments=(
    "baseline"
    "relu_activation"
    "leaky_relu_activation"
    "mse_loss"
    "relu_mse_combo"
)

# 批量运行实验
for exp_name in "${experiments[@]}"; do
    echo "Running experiment: $exp_name"
    
    case $exp_name in
        "baseline")
            python main.py --model TranAD_QC --dataset my_data --experiment_name "$exp_name" --visualize
            ;;
        "relu_activation")
            python main.py --model TranAD_QC --dataset my_data --experiment_name "$exp_name" --activation_type relu --visualize
            ;;
        "leaky_relu_activation")
            python main.py --model TranAD_QC --dataset my_data --experiment_name "$exp_name" --activation_type leaky_relu --visualize
            ;;
        "mse_loss")
            python main.py --model TranAD_QC --dataset my_data --experiment_name "$exp_name" --loss_type mse --visualize
            ;;
        "relu_mse_combo")
            python main.py --model TranAD_QC --dataset my_data --experiment_name "$exp_name" --activation_type relu --loss_type mse --visualize
            ;;
    esac
    
    echo "Completed experiment: $exp_name"
    echo "---"
done
```

## 5-Fold交叉验证

### 1. 交叉验证说明

项目使用5-fold分层交叉验证来评估模型性能：
- 将数据分成5个相等的部分
- 每个fold保持正常/异常样本的比例
- 进行5轮训练和测试
- 计算平均性能指标

### 2. 交叉验证可视化

```bash
# 生成5-fold交叉验证可视化
python main.py --model TranAD_QC --dataset my_data --visualize --vis_n_folds 5
```

## 文件结构

```
TranAD-main/
├── main.py                 # 主程序
├── src/                    # 源代码
│   ├── models.py          # 模型定义
│   ├── parser.py          # 参数解析
│   ├── utils.py           # 工具函数
│   └── ...
├── data/                   # 数据目录
│   ├── my_data/           # 自定义数据
│   ├── MBA/               # MBA数据集
│   └── ...
├── checkpoints/            # 模型检查点
│   └── {experiment_name}/ # 实验特定的检查点
├── plots/                  # 可视化结果
│   └── classification/    # 分类结果可视化
│       └── {experiment_name}/ # 实验特定的可视化
├── processed/              # 预处理后的数据
└── visualization.py        # 可视化模块
```

## 常见问题

### 1. 内存不足

```bash
# 减少批次大小
python main.py --model TranAD_QC --dataset my_data --num_epochs 5
```

### 2. 训练时间过长

```bash
# 使用预训练模型
python main.py --model TranAD_QC --dataset my_data --test
```

### 3. 可视化失败

```bash
# 检查依赖
pip install matplotlib seaborn scikit-learn
```

## 性能优化

### 1. GPU加速

确保安装了CUDA版本的PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 数据预处理

```bash
# 预处理数据以提高训练速度
python preprocess_2_mydata.py
```

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请提交Issue或联系项目维护者。

---

## 快速参考

### 你的训练命令

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

这个命令会：
1. 训练TranAD_QC模型
2. 使用learnable_cosine激活函数
3. 使用complex_aware损失函数
4. 训练分类头（v2版本）
5. 重新训练10个epoch
6. 生成完整的可视化报告
7. 保存为"CV-FCNN_learnable_cosine"实验