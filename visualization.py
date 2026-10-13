#!/usr/bin/env python3
"""
分类结果可视化模块
支持SVM风格的分类边界可视化、多fold测试集分析、特征降维等功能
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import pandas as pd
import torch
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 强制使用英文，避免字体问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 设置matplotlib参数
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# 全局变量：是否使用中文
USE_CHINESE = False  # 强制使用英文
chinese_font = None  # 不使用中文字体

class ClassificationVisualizer:
    """分类结果可视化器"""
    
    def __init__(self, save_dir: str = "plots/classification"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def extract_features_and_labels(self, model, test_windows, test_labels, mode='complex'):
        """从模型中提取特征和标签"""
        model.eval()
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        features_list = []
        scores_list = []
        
        with torch.no_grad():
            bs = model.batch
            data_x = torch.tensor(test_windows, dtype=dtype, device=device)
            from torch.utils.data import TensorDataset, DataLoader
            dataset = TensorDataset(data_x, data_x)
            dl = DataLoader(dataset, batch_size=bs, shuffle=False)
            
            for d, _ in dl:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, d.shape[0], test_windows.shape[-1])
                
                # 提取误差特征
                error_feat = model.extract_error_features(window, elem, mode=mode)
                features_list.append(error_feat.detach().cpu())
                
                # 获取分类分数
                logits = model.predict_scores(window, elem, mode=mode)
                scores_list.append(logits.detach().cpu())
        
        # 合并所有batch的结果
        features = torch.cat(features_list, dim=0).numpy()
        scores = torch.cat(scores_list, dim=0).numpy()
        
        # 行级标签
        labels = (test_labels.sum(axis=1) > 0).astype(np.float64)
        
        return features, scores, labels
    
    def reduce_dimensions(self, features: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """降维到2D用于可视化"""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 't-sne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=min(30, features.shape[0]//4))
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        return reducer.fit_transform(features)
    
    def plot_classification_boundary(self, features_2d: np.ndarray, labels: np.ndarray, 
                                   scores: np.ndarray, title: str = "分类边界可视化",
                                   save_name: str = "classification_boundary.png"):
        """绘制SVM风格的分类边界"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 创建网格用于绘制决策边界
        h = 0.02  # 网格步长
        x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
        y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 使用logistic回归拟合决策边界
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
        clf.fit(features_2d, labels)
        
        # 预测网格点
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict_proba(grid_points)[:, 1]
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        ax1.contourf(xx, yy, Z, levels=50, alpha=0.4, cmap='RdYlBu_r')
        contours = ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=3)
        ax1.clabel(contours, inline=True, fontsize=10, fmt='Decision Boundary')
        
        # 绘制数据点
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        # 使用英文标签
        normal_label = f'Normal ({normal_mask.sum()})'
        anomaly_label = f'Anomaly ({anomaly_mask.sum()})'
        
        ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                   c='#2E86AB', marker='o', s=60, alpha=0.8, label=normal_label, edgecolors='white', linewidth=0.5)
        ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1], 
                   c='#A23B72', marker='^', s=60, alpha=0.8, label=anomaly_label, edgecolors='white', linewidth=0.5)
        
        ax1.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Second Principal Component', fontsize=12, fontweight='bold')
        ax1.set_title('Classification Boundary and Data Distribution', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 第二个子图：分类置信度分布
        # 将分数转换为概率
        probs = 1 / (1 + np.exp(-scores))
        
        ax2.hist(probs[normal_mask], bins=30, alpha=0.7, color='#2E86AB', 
                label=f'Normal (μ={probs[normal_mask].mean():.3f})', density=True, edgecolor='white', linewidth=0.5)
        ax2.hist(probs[anomaly_mask], bins=30, alpha=0.7, color='#A23B72', 
                label=f'Anomaly (μ={probs[anomaly_mask].mean():.3f})', density=True, edgecolor='white', linewidth=0.5)
        
        # 添加阈值线
        threshold = 0.5
        ax2.axvline(threshold, color='black', linestyle='--', linewidth=3, 
                   label=f'Decision Threshold ({threshold})')
        
        ax2.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax2.set_title('Classification Score Distribution', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Classification boundary plot saved to: {save_path}")
    
    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, 
                            title: str = "Confusion Matrix", save_name: str = "confusion_matrix.png"):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, predictions)
        
        # 创建更大的图形，为指标框留出空间
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                      gridspec_kw={'width_ratios': [3, 1]})
        
        # 主混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
                   ax=ax1, cbar_kws={'label': 'Sample Count'})
        
        ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # 计算性能指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        rec = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        # 在右侧创建指标框
        ax2.axis('off')
        
        # 创建美观的指标显示
        metrics_data = [
            ['Accuracy', f'{acc:.3f}'],
            ['Precision', f'{prec:.3f}'],
            ['Recall', f'{rec:.3f}'],
            ['F1-Score', f'{f1:.3f}']
        ]
        
        # 创建表格样式的指标显示
        table = ax2.table(cellText=metrics_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.3, 0.8, 0.4])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)
        
        # 设置表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置数据行样式
        for i in range(1, 5):
            for j in range(2):
                if j == 0:  # 指标名称列
                    table[(i, j)].set_facecolor('#E8F5E8')
                    table[(i, j)].set_text_props(weight='bold')
                else:  # 数值列
                    table[(i, j)].set_facecolor('#F0F8FF')
                    table[(i, j)].set_text_props(weight='bold', color='#2E86AB')
        
        # 添加样本统计信息
        total_samples = len(labels)
        normal_samples = (labels == 0).sum()
        anomaly_samples = (labels == 1).sum()
        
        stats_text = f"""
Sample Statistics:
• Total: {total_samples}
• Normal: {normal_samples} ({normal_samples/total_samples*100:.1f}%)
• Anomaly: {anomaly_samples} ({anomaly_samples/total_samples*100:.1f}%)
        """
        
        ax2.text(0.1, 0.1, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8DC", alpha=0.8),
                verticalalignment='bottom')
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to: {save_path}")
    
    def plot_roc_curve(self, labels: np.ndarray, scores: np.ndarray, 
                      title: str = "ROC Curve", save_name: str = "roc_curve.png"):
        """绘制ROC曲线"""
        # 将logits转换为概率
        probs = 1 / (1 + np.exp(-scores))
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制ROC曲线
        ax.plot(fpr, tpr, color='#2E86AB', lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#A23B72', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.500)')
        
        # 填充曲线下方区域
        ax.fill_between(fpr, tpr, alpha=0.3, color='#2E86AB')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 添加AUC值标注
        ax.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E8", alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"ROC curve saved to: {save_path}")
        
        return roc_auc
    
    def plot_feature_importance(self, features: np.ndarray, labels: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               title: str = "特征重要性", save_name: str = "feature_importance.png"):
        """绘制特征重要性"""
        from sklearn.ensemble import RandomForestClassifier
        
        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features, labels)
        importance = rf.feature_importances_
        
        # 选择前20个最重要的特征
        n_features = min(20, len(importance))
        indices = np.argsort(importance)[::-1][:n_features]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(n_features), importance[indices], color='steelblue', alpha=0.7)
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title(title)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # 在柱子上添加数值
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Feature importance plot saved to: {save_path}")
    
    def multi_fold_visualization(self, model, test_windows, test_labels, 
                                n_folds: int = 5, mode: str = 'complex'):
        """多fold交叉验证可视化"""
        print(f"\n=== 开始{n_folds}折交叉验证可视化 ===")
        
        # 行级标签
        labels = (test_labels.sum(axis=1) > 0).astype(np.float64)
        
        # 创建分层K折
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_features_2d = []
        all_labels = []
        all_scores = []
        all_folds = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(test_windows, labels)):
            print(f"\n处理第 {fold+1} 折...")
            
            # 当前fold的测试数据
            fold_windows = test_windows[test_idx]
            fold_labels = test_labels[test_idx]
            
            # 提取特征和分数
            features, scores, fold_labels_row = self.extract_features_and_labels(
                model, fold_windows, fold_labels, mode=mode)
            
            # 降维到2D
            features_2d = self.reduce_dimensions(features, method='pca')
            
            # 保存结果
            fold_results.append({
                'fold': fold + 1,
                'features': features,
                'features_2d': features_2d,
                'labels': fold_labels_row,
                'scores': scores,
                'n_normal': (fold_labels_row == 0).sum(),
                'n_anomaly': (fold_labels_row == 1).sum()
            })
            
            # 合并到总体结果
            all_features_2d.append(features_2d)
            all_labels.append(fold_labels_row)
            all_scores.append(scores)
            all_folds.append(np.full(len(fold_labels_row), fold + 1))
        
        # 绘制每个fold的结果
        self.plot_multi_fold_comparison(fold_results)
        
        # 绘制总体结果
        all_features_2d = np.vstack(all_features_2d)
        all_labels = np.concatenate(all_labels)
        all_scores = np.concatenate(all_scores)
        all_folds = np.concatenate(all_folds)
        
        self.plot_overall_results(all_features_2d, all_labels, all_scores, all_folds, n_folds)
        
        return fold_results
    
    def plot_multi_fold_comparison(self, fold_results: List[Dict]):
        """绘制多fold比较图"""
        n_folds = len(fold_results)
        fig, axes = plt.subplots(2, (n_folds + 1) // 2, figsize=(15, 10))
        if n_folds == 1:
            axes = [axes]
        elif n_folds <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(fold_results):
            ax = axes[i] if len(axes.shape) == 1 else axes[i]
            
            features_2d = result['features_2d']
            labels = result['labels']
            
            # 绘制散点图
            normal_mask = labels == 0
            anomaly_mask = labels == 1
            
            ax.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                      c='blue', marker='o', s=30, alpha=0.6, label=f'正常 ({result["n_normal"]})')
            ax.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1], 
                      c='red', marker='^', s=30, alpha=0.6, label=f'异常 ({result["n_anomaly"]})')
            
            # 添加决策边界
            if len(np.unique(labels)) > 1:  # 确保有两个类
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(random_state=42)
                clf.fit(features_2d, labels)
                
                # 创建网格
                h = 0.02
                x_min, x_max = features_2d[:, 0].min() - 0.5, features_2d[:, 0].max() + 0.5
                y_min, y_max = features_2d[:, 1].min() - 0.5, features_2d[:, 1].max() + 0.5
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))
                
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
                Z = Z.reshape(xx.shape)
                
                ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=1)
            
            ax.set_title(f'Fold {result["fold"]}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的subplot
        for j in range(len(fold_results), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Multi-Fold Classification Results Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "multi_fold_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Multi-fold comparison plot saved to: {save_path}")
    
    def plot_overall_results(self, features_2d: np.ndarray, labels: np.ndarray, 
                           scores: np.ndarray, folds: np.ndarray, n_folds: int):
        """绘制总体结果图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 总体分类边界
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        # 决策边界
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=42)
        clf.fit(features_2d, labels)
        
        h = 0.02
        x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
        y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        
        ax1.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
        ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                   c='blue', marker='o', s=20, alpha=0.6, label=f'正常 ({normal_mask.sum()})')
        ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1], 
                   c='red', marker='^', s=20, alpha=0.6, label=f'异常 ({anomaly_mask.sum()})')
        
        ax1.set_title('Overall Classification Boundary')
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 按fold着色的散点图
        colors = plt.cm.Set3(np.linspace(0, 1, n_folds))
        for fold in range(1, n_folds + 1):
            fold_mask = folds == fold
            ax2.scatter(features_2d[fold_mask, 0], features_2d[fold_mask, 1], 
                       c=[colors[fold-1]], s=20, alpha=0.7, label=f'Fold {fold}')
        
        ax2.set_title('Data Distribution by Fold')
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. ROC曲线
        probs = 1 / (1 + np.exp(-scores))
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 分类分数分布
        ax4.hist(probs[normal_mask], bins=30, alpha=0.7, color='blue', 
                label=f'Normal (μ={probs[normal_mask].mean():.3f})', density=True)
        ax4.hist(probs[anomaly_mask], bins=30, alpha=0.7, color='red', 
                label=f'Anomaly (μ={probs[anomaly_mask].mean():.3f})', density=True)
        ax4.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax4.set_xlabel('Prediction Probability')
        ax4.set_ylabel('Density')
        ax4.set_title('Classification Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{n_folds}-Fold Cross-Validation Overall Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, "overall_results.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Overall results plot saved to: {save_path}")
    
    def generate_full_report(self, model, test_windows, test_labels, 
                           model_name: str = "TranAD_QC", n_folds: int = 5):
        """生成完整的可视化报告"""
        print(f"\n{'='*60}")
        print(f"生成 {model_name} 分类结果可视化报告")
        print(f"{'='*60}")
        
        # 创建模型特定的保存目录
        model_save_dir = os.path.join(self.save_dir, model_name.replace('/', '_'))
        os.makedirs(model_save_dir, exist_ok=True)
        original_save_dir = self.save_dir
        self.save_dir = model_save_dir
        
        try:
            # 1. 总体特征提取和可视化
            print("\n1. 提取特征并生成总体可视化...")
            features, scores, labels = self.extract_features_and_labels(
                model, test_windows, test_labels, mode='complex')
            
            # 降维
            features_2d_pca = self.reduce_dimensions(features, method='pca')
            features_2d_tsne = self.reduce_dimensions(features, method='t-sne')
            
            # 生成各种可视化
            self.plot_classification_boundary(
                features_2d_pca, labels, scores, 
                title=f"{model_name} - PCA降维分类边界",
                save_name="classification_boundary_pca.png")
            
            self.plot_classification_boundary(
                features_2d_tsne, labels, scores, 
                title=f"{model_name} - t-SNE降维分类边界",
                save_name="classification_boundary_tsne.png")
            
            # 2. 生成评估图表
            print("\n2. 生成评估图表...")
            predictions = (scores > 0).astype(int)  # 使用0作为阈值
            
            self.plot_confusion_matrix(
                labels, predictions, 
                title=f"{model_name} - 混淆矩阵",
                save_name="confusion_matrix.png")
            
            roc_auc = self.plot_roc_curve(
                labels, scores, 
                title=f"{model_name} - ROC曲线",
                save_name="roc_curve.png")
            
            # 生成F1-score分析
            print("\n3. 生成F1-score分析...")
            f1_results = self.plot_f1_score_analysis(
                labels, scores,
                title=f"{model_name} - F1-Score Analysis",
                save_name="f1_score_analysis.png")
            
            self.plot_feature_importance(
                features, labels,
                title=f"{model_name} - 特征重要性",
                save_name="feature_importance.png")
            
            # 4. 多fold交叉验证
            print(f"\n4. 进行{n_folds}折交叉验证可视化...")
            fold_results = self.multi_fold_visualization(
                model, test_windows, test_labels, n_folds=n_folds)
            
            # 5. 生成统计报告
            print("\n5. 生成统计报告...")
            self.generate_statistics_report(labels, predictions, scores, fold_results, model_name)
            
            print(f"\n✓ 完整可视化报告已生成到: {model_save_dir}")
            
        finally:
            # 恢复原始保存目录
            self.save_dir = original_save_dir
    
    def generate_statistics_report(self, labels, predictions, scores, fold_results, model_name):
        """生成统计报告"""
        from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
        
        # 总体统计
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions)
        rec = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        # 计算AUC
        probs = 1 / (1 + np.exp(-scores))
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(labels, probs)
        
        # 按fold统计
        fold_stats = []
        for result in fold_results:
            fold_labels = result['labels']
            fold_scores = result['scores']
            fold_preds = (fold_scores > 0).astype(int)
            
            fold_acc = accuracy_score(fold_labels, fold_preds)
            fold_f1 = f1_score(fold_labels, fold_preds) if len(np.unique(fold_labels)) > 1 else 0
            
            fold_stats.append({
                'fold': result['fold'],
                'accuracy': fold_acc,
                'f1_score': fold_f1,
                'n_normal': result['n_normal'],
                'n_anomaly': result['n_anomaly']
            })
        
        # 生成报告文本
        report = f"""
{model_name} Classification Results Report
{'='*60}

Overall Performance:
- Accuracy: {acc:.4f}
- Precision: {prec:.4f}
- Recall: {rec:.4f}
- F1-Score: {f1:.4f}
- AUC Score: {auc_score:.4f}

Data Distribution:
- Normal samples: {(labels == 0).sum()} ({(labels == 0).mean()*100:.1f}%)
- Anomaly samples: {(labels == 1).sum()} ({(labels == 1).mean()*100:.1f}%)
- Total samples: {len(labels)}

Performance by Fold:
"""
        
        for stat in fold_stats:
            report += f"- Fold {stat['fold']}: Accuracy={stat['accuracy']:.4f}, F1={stat['f1_score']:.4f}, "
            report += f"Normal={stat['n_normal']}, Anomaly={stat['n_anomaly']}\n"
        
        # 保存报告
        report_path = os.path.join(self.save_dir, "classification_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Statistical report saved to: {report_path}")
        print("\n" + "="*40)
        print("Main Performance Metrics:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC Score: {auc_score:.4f}")
        print("="*40)
    
    def plot_f1_score_analysis(self, labels: np.ndarray, scores: np.ndarray, 
                              title: str = "F1-Score Analysis", save_name: str = "f1_score_analysis.png"):
        """绘制F1-score分析图，包括不同阈值下的F1-score变化"""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # 生成不同阈值下的F1-score
        thresholds = np.linspace(0.1, 0.9, 50)
        f1_scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            predictions = (scores > threshold).astype(int)
            if len(np.unique(predictions)) > 1:  # 确保有正负样本
                f1 = f1_score(labels, predictions)
                prec = precision_score(labels, predictions)
                rec = recall_score(labels, predictions)
            else:
                f1 = prec = rec = 0.0
            
            f1_scores.append(f1)
            precisions.append(prec)
            recalls.append(rec)
        
        # 创建图形
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. F1-score vs 阈值
        ax1.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1-Score')
        ax1.set_xlabel('Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax1.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 找到最佳F1-score
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        ax1.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Best F1: {best_f1:.3f} @ {best_threshold:.3f}')
        ax1.legend()
        
        # 2. Precision, Recall, F1 vs 阈值
        ax2.plot(thresholds, precisions, 'g-', linewidth=2, label='Precision')
        ax2.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall')
        ax2.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1-Score')
        ax2.set_xlabel('Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax2.set_title('Precision, Recall, F1-Score vs Threshold', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. F1-score分布直方图
        predictions_best = (scores > best_threshold).astype(int)
        f1_best = f1_score(labels, predictions_best)
        
        # 计算不同fold的F1-score（如果数据足够大）
        if len(labels) > 100:
            from sklearn.model_selection import StratifiedKFold
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_f1_scores = []
            
            for train_idx, val_idx in kf.split(scores, labels):
                val_scores = scores[val_idx]
                val_labels = labels[val_idx]
                val_preds = (val_scores > best_threshold).astype(int)
                if len(np.unique(val_preds)) > 1:
                    fold_f1 = f1_score(val_labels, val_preds)
                    fold_f1_scores.append(fold_f1)
            
            if fold_f1_scores:
                ax3.hist(fold_f1_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(x=np.mean(fold_f1_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(fold_f1_scores):.3f}')
                ax3.set_xlabel('F1-Score', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax3.set_title('F1-Score Distribution (5-Fold CV)', fontsize=14, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
        else:
            # 如果数据太小，显示F1-score的置信区间
            ax3.text(0.5, 0.5, f'F1-Score: {f1_best:.3f}\n(Data too small for CV)', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            ax3.set_xlim(0, 1)
            ax3.set_ylim(0, 1)
            ax3.set_title('F1-Score Summary', fontsize=14, fontweight='bold')
            ax3.axis('off')
        
        # 4. 性能指标对比
        from sklearn.metrics import accuracy_score
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [
            accuracy_score(labels, predictions_best),
            precision_score(labels, predictions_best),
            recall_score(labels, predictions_best),
            f1_best
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax4.set_title('Performance Metrics Summary', fontsize=14, fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"F1-score analysis plot saved to: {save_path}")
        
        return {
            'best_threshold': best_threshold,
            'best_f1_score': best_f1,
            'f1_scores': f1_scores,
            'thresholds': thresholds
        }
    
    def plot_f1_comparison(self, results_dict: Dict[str, Dict], 
                          title: str = "F1-Score Comparison", save_name: str = "f1_comparison.png"):
        """比较不同模型或配置的F1-score"""
        models = list(results_dict.keys())
        f1_scores = [results_dict[model]['f1_score'] for model in models]
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 柱状图比较
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax1.bar(models, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
        ax1.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 旋转x轴标签
        ax1.tick_params(axis='x', rotation=45)
        
        # 在柱状图上添加数值标签
        for bar, value in zip(bars, f1_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. 雷达图比较（如果有多个指标）
        if len(results_dict[models[0]]) > 1:
            metrics = ['F1-Score', 'Precision', 'Recall', 'Accuracy']
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            
            ax2 = plt.subplot(122, projection='polar')
            
            for i, model in enumerate(models):
                values = [
                    results_dict[model].get('f1_score', 0),
                    results_dict[model].get('precision', 0),
                    results_dict[model].get('recall', 0),
                    results_dict[model].get('accuracy', 0)
                ]
                values += values[:1]  # 闭合图形
                
                ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
                ax2.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(metrics)
            ax2.set_ylim(0, 1)
            ax2.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
            ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax2.grid(True)
        else:
            # 如果没有多个指标，显示简单的排名
            sorted_models = sorted(zip(models, f1_scores), key=lambda x: x[1], reverse=True)
            ax2.axis('off')
            
            ranking_text = "F1-Score Ranking:\n\n"
            for i, (model, score) in enumerate(sorted_models, 1):
                ranking_text += f"{i}. {model}: {score:.3f}\n"
            
            ax2.text(0.1, 0.5, ranking_text, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                    verticalalignment='center')
            ax2.set_title('Model Ranking', fontsize=14, fontweight='bold')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"F1-score comparison plot saved to: {save_path}")


def visualize_classification_results(model, test_windows, test_labels, 
                                   model_name: str = "TranAD_QC", n_folds: int = 5,
                                   save_dir: str = "plots/classification"):
    """便捷函数：生成完整的分类结果可视化"""
    visualizer = ClassificationVisualizer(save_dir=save_dir)
    visualizer.generate_full_report(model, test_windows, test_labels, model_name, n_folds)


if __name__ == "__main__":
    print("分类结果可视化模块")
    print("使用方法：")
    print("from visualization import visualize_classification_results")
    print("visualize_classification_results(model, test_windows, test_labels, model_name='your_model')")
