# 实验名称参数使用指南

## 新增参数

在命令行参数中新增了 `--experiment_name` 参数，用于指定实验的自定义名称。

## 使用方法

### 1. 基本用法

```bash
# 使用自定义实验名称
python main.py --model TranAD_QC --dataset my_data --experiment_name "my_custom_experiment"

# 不使用实验名称（使用默认命名规则）
python main.py --model TranAD_QC --dataset my_data
```

### 2. 与可视化结合使用

```bash
# 生成可视化并使用自定义名称
python main.py --model TranAD_QC --dataset my_data --experiment_name "experiment_v1" --visualize

# 指定可视化保存目录
python main.py --model TranAD_QC --dataset my_data --experiment_name "experiment_v1" --visualize --vis_save_dir "plots/my_experiments"
```

### 3. 与消融研究结合使用

```bash
# 不同激活函数的实验
python main.py --model TranAD_QC --dataset my_data --experiment_name "relu_activation" --activation_type relu --visualize

python main.py --model TranAD_QC --dataset my_data --experiment_name "leaky_relu_activation" --activation_type leaky_relu --visualize

# 不同损失函数的实验
python main.py --model TranAD_QC --dataset my_data --experiment_name "mse_loss" --loss_type mse --visualize
```

### 4. 训练和测试

```bash
# 训练模型
python main.py --model TranAD_QC --dataset my_data --experiment_name "baseline_model"

# 测试已训练的模型
python main.py --model TranAD_QC --dataset my_data --experiment_name "baseline_model" --test
```

## 命名规则

### 1. 使用自定义名称时
- 所有输出（checkpoints、可视化、日志）都使用指定的名称
- 例如：`--experiment_name "my_experiment"` 会生成：
  - Checkpoint: `checkpoints/my_experiment/model.ckpt`
  - 可视化: `plots/classification/my_experiment/`

### 2. 使用默认命名时
- 基础格式：`{model_name}_{dataset}`
- 对于TranAD_QC，会添加消融研究参数：
  - 例如：`TranAD_QC_my_data_relu_mse`（如果使用了非默认的relu激活和mse损失）

## 实际应用场景

### 1. 消融研究
```bash
# 实验1：默认配置
python main.py --model TranAD_QC --dataset my_data --experiment_name "baseline" --visualize

# 实验2：使用ReLU激活
python main.py --model TranAD_QC --dataset my_data --experiment_name "relu_activation" --activation_type relu --visualize

# 实验3：使用MSE损失
python main.py --model TranAD_QC --dataset my_data --experiment_name "mse_loss" --loss_type mse --visualize

# 实验4：组合配置
python main.py --model TranAD_QC --dataset my_data --experiment_name "relu_mse_combo" --activation_type relu --loss_type mse --visualize
```

### 2. 不同数据集实验
```bash
# 在my_data上的实验
python main.py --model TranAD_QC --dataset my_data --experiment_name "my_data_experiment" --visualize

# 在MBA上的实验
python main.py --model TranAD_QC --dataset MBA --experiment_name "mba_experiment" --visualize
```

### 3. 不同模型比较
```bash
# TranAD_QC实验
python main.py --model TranAD_QC --dataset my_data --experiment_name "tranad_qc_experiment" --visualize

# TranAD_C实验
python main.py --model TranAD_C --dataset my_data --experiment_name "tranad_c_experiment" --visualize
```

## 输出文件结构

使用 `--experiment_name "my_experiment"` 时，会生成以下文件结构：

```
checkpoints/
└── my_experiment/
    └── model.ckpt

plots/
└── classification/
    └── my_experiment/
        ├── classification_boundary_pca.png
        ├── classification_boundary_tsne.png
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── f1_score_analysis.png
        ├── feature_importance.png
        ├── multi_fold_analysis.png
        └── classification_report.txt
```

## 注意事项

1. **实验名称唯一性**：确保不同的实验使用不同的名称，避免覆盖之前的实验结果
2. **命名规范**：建议使用有意义的名称，如 `baseline_v1`、`relu_activation`、`mse_loss` 等
3. **路径兼容性**：实验名称会用于文件路径，避免使用特殊字符
4. **向后兼容**：如果不指定 `--experiment_name`，系统会自动生成名称，保持向后兼容

## 批量实验脚本示例

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

这样，你就可以轻松地管理和比较不同的实验结果了！
