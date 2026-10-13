#!/usr/bin/env python3
"""
修复的分类结果可视化模块
解决中文字体问题和PCA降维问题
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

# 设置matplotlib为非交互式后端，避免字体问题
import matplotlib
matplotlib.use('Agg')

# 设置绘图风格，使用英文避免字体问题
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ClassificationVisualizerFixed:
    """修复的分类结果可视化器"""
    
    def __init__(self, save_dir: str = "plots/classification"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
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
        
        print(f"[DEBUG] Features shape: {features.shape}")
        print(f"[DEBUG] Features range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"[DEBUG] Features std: {features.std():.4f}")
        
        return features, scores, labels
    
    def reduce_dimensions(self, features: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
        """降维到2D用于可视化，增加调试信息"""
        print(f"[DEBUG] 开始{method.upper()}降维，输入形状: {features.shape}")
        
        # 检查特征是否有效
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            print("[WARNING] 特征中包含NaN或Inf值，进行清理...")
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 检查特征方差
        feature_std = np.std(features, axis=0)
        zero_var_features = np.sum(feature_std < 1e-8)
        if zero_var_features > 0:
            print(f"[WARNING] {zero_var_features} 个特征方差接近零")
        
        if method == 'pca':
            # 标准化特征
            features_normalized = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
            
            reducer = PCA(n_components=n_components, random_state=42)
            reduced_features = reducer.fit_transform(features_normalized)
            
            # 打印PCA信息
            explained_variance = reducer.explained_variance_ratio_
            print(f"[DEBUG] PCA解释方差比: {explained_variance}")
            print(f"[DEBUG] 累计解释方差: {np.sum(explained_variance):.4f}")
            
        elif method == 't-sne':
            # t-SNE对数据规模敏感，如果数据太多就采样
            if features.shape[0] > 1000:
                indices = np.random.choice(features.shape[0], 1000, replace=False)
                features_sample = features[indices]
                print(f"[DEBUG] t-SNE采样到{len(indices)}个样本")
            else:
                features_sample = features
            
            # 先用PCA降维到合适维度（t-SNE前预处理）
            if features_sample.shape[1] > 50:
                pca_pre = PCA(n_components=50, random_state=42)
                features_sample = pca_pre.fit_transform(features_sample)
                print(f"[DEBUG] t-SNE前PCA预处理到{features_sample.shape[1]}维")
            
            perplexity = min(30, features_sample.shape[0] // 4)
            reducer = TSNE(n_components=n_components, random_state=42, 
                          perplexity=perplexity, learning_rate='auto', init='pca')
            
            if features.shape[0] > 1000:
                reduced_sample = reducer.fit_transform(features_sample)
                # 为剩余数据插值（简单方法）
                reduced_features = np.zeros((features.shape[0], n_components))
                reduced_features[indices] = reduced_sample
                # 对未采样的点使用最近邻插值
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=5).fit(features[indices])
                for i in range(features.shape[0]):
                    if i not in indices:
                        _, neighbors = nbrs.kneighbors([features[i]])
                        reduced_features[i] = np.mean(reduced_sample[neighbors[0]], axis=0)
            else:
                reduced_features = reducer.fit_transform(features_sample)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
        
        print(f"[DEBUG] {method.upper()}降维后形状: {reduced_features.shape}")
        print(f"[DEBUG] {method.upper()}降维后范围: [{reduced_features.min():.4f}, {reduced_features.max():.4f}]")
        
        return reduced_features
    
    def plot_classification_boundary_fixed(self, features_2d: np.ndarray, labels: np.ndarray, 
                                         scores: np.ndarray, title: str = "Classification Boundary",
                                         save_name: str = "classification_boundary.png"):
        """修复的分类边界绘制函数"""
        print(f"[DEBUG] 绘制分类边界: {features_2d.shape}, unique labels: {np.unique(labels)}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 检查数据有效性
        if len(np.unique(labels)) < 2:
            print("[WARNING] 只有一个类别，无法绘制决策边界")
            ax1.text(0.5, 0.5, 'Only one class present', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(title)
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, save_name)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # 绘制数据点
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        # 检查数据点分布
        print(f"[DEBUG] 正常样本: {normal_mask.sum()}, 异常样本: {anomaly_mask.sum()}")
        print(f"[DEBUG] 特征范围 X: [{features_2d[:, 0].min():.4f}, {features_2d[:, 0].max():.4f}]")
        print(f"[DEBUG] 特征范围 Y: [{features_2d[:, 1].min():.4f}, {features_2d[:, 1].max():.4f}]")
        
        # 如果数据点太集中，扩展显示范围
        x_range = features_2d[:, 0].max() - features_2d[:, 0].min()
        y_range = features_2d[:, 1].max() - features_2d[:, 1].min()
        
        if x_range < 1e-6 or y_range < 1e-6:
            print("[WARNING] 数据点过于集中，可能降维有问题")
            # 人工扩展范围用于可视化
            center_x, center_y = np.mean(features_2d, axis=0)
            expand_range = 1.0
            x_min, x_max = center_x - expand_range, center_x + expand_range
            y_min, y_max = center_y - expand_range, center_y + expand_range
        else:
            margin = max(x_range, y_range) * 0.1
            x_min, x_max = features_2d[:, 0].min() - margin, features_2d[:, 0].max() + margin
            y_min, y_max = features_2d[:, 1].min() - margin, features_2d[:, 1].max() + margin
        
        # 创建网格用于绘制决策边界
        h = max((x_max - x_min) / 100, (y_max - y_min) / 100)
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # 使用logistic回归拟合决策边界
        from sklearn.linear_model import LogisticRegression
        try:
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(features_2d, labels)
            
            # 预测网格点
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.predict_proba(grid_points)[:, 1]
            Z = Z.reshape(xx.shape)
            
            # 绘制决策边界
            contourf = ax1.contourf(xx, yy, Z, levels=20, alpha=0.6, cmap='RdYlBu')
            contours = ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
            
        except Exception as e:
            print(f"[WARNING] 决策边界绘制失败: {e}")
        
        # 绘制数据点
        ax1.scatter(features_2d[normal_mask, 0], features_2d[normal_mask, 1], 
                   c='blue', marker='o', s=50, alpha=0.7, label=f'Normal ({normal_mask.sum()})')
        ax1.scatter(features_2d[anomaly_mask, 0], features_2d[anomaly_mask, 1], 
                   c='red', marker='^', s=50, alpha=0.7, label=f'Anomaly ({anomaly_mask.sum()})')
        
        ax1.set_xlabel('Component 1')
        ax1.set_ylabel('Component 2')
        ax1.set_title('Classification Boundary')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 第二个子图：分类置信度分布
        probs = 1 / (1 + np.exp(-scores))
        
        ax2.hist(probs[normal_mask], bins=30, alpha=0.7, color='blue', 
                label=f'Normal (μ={probs[normal_mask].mean():.3f})', density=True)
        ax2.hist(probs[anomaly_mask], bins=30, alpha=0.7, color='red', 
                label=f'Anomaly (μ={probs[anomaly_mask].mean():.3f})', density=True)
        
        threshold = 0.5
        ax2.axvline(threshold, color='black', linestyle='--', linewidth=2, 
                   label=f'Threshold ({threshold})')
        
        ax2.set_xlabel('Prediction Probability')
        ax2.set_ylabel('Density')
        ax2.set_title('Classification Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形释放内存
        print(f"Classification boundary plot saved to: {save_path}")
    
    def compare_pca_tsne(self, model, test_windows, test_labels, model_name: str = "TranAD_QC"):
        """比较PCA和t-SNE的降维效果"""
        print(f"\n{'='*60}")
        print(f"Comparing PCA vs t-SNE for {model_name}")
        print(f"{'='*60}")
        
        # 提取特征
        features, scores, labels = self.extract_features_and_labels(
            model, test_windows, test_labels, mode='complex')
        
        # PCA降维
        print("\n--- PCA Analysis ---")
        features_2d_pca = self.reduce_dimensions(features, method='pca')
        
        # t-SNE降维
        print("\n--- t-SNE Analysis ---")
        features_2d_tsne = self.reduce_dimensions(features, method='t-sne')
        
        # 绘制比较图
        self.plot_classification_boundary_fixed(
            features_2d_pca, labels, scores,
            title=f"{model_name} - PCA Dimensionality Reduction",
            save_name="classification_boundary_pca_fixed.png"
        )
        
        self.plot_classification_boundary_fixed(
            features_2d_tsne, labels, scores,
            title=f"{model_name} - t-SNE Dimensionality Reduction", 
            save_name="classification_boundary_tsne_fixed.png"
        )
        
        # 分析结果
        self._analyze_dimensionality_reduction(features_2d_pca, features_2d_tsne, labels, model_name)
    
    def _analyze_dimensionality_reduction(self, features_pca, features_tsne, labels, model_name):
        """分析降维结果"""
        print(f"\n{'='*40}")
        print("Dimensionality Reduction Analysis")
        print(f"{'='*40}")
        
        # PCA分析
        pca_spread = np.std(features_pca, axis=0)
        print(f"PCA spread (std): Component1={pca_spread[0]:.4f}, Component2={pca_spread[1]:.4f}")
        
        pca_range = np.ptp(features_pca, axis=0)  # peak-to-peak range
        print(f"PCA range: Component1={pca_range[0]:.4f}, Component2={pca_range[1]:.4f}")
        
        # t-SNE分析
        tsne_spread = np.std(features_tsne, axis=0)
        print(f"t-SNE spread (std): Component1={tsne_spread[0]:.4f}, Component2={tsne_spread[1]:.4f}")
        
        tsne_range = np.ptp(features_tsne, axis=0)
        print(f"t-SNE range: Component1={tsne_range[0]:.4f}, Component2={tsne_range[1]:.4f}")
        
        # 类别分离度分析
        normal_mask = labels == 0
        anomaly_mask = labels == 1
        
        if np.sum(normal_mask) > 0 and np.sum(anomaly_mask) > 0:
            # PCA类别分离
            pca_normal_center = np.mean(features_pca[normal_mask], axis=0)
            pca_anomaly_center = np.mean(features_pca[anomaly_mask], axis=0)
            pca_separation = np.linalg.norm(pca_normal_center - pca_anomaly_center)
            
            # t-SNE类别分离
            tsne_normal_center = np.mean(features_tsne[normal_mask], axis=0)
            tsne_anomaly_center = np.mean(features_tsne[anomaly_mask], axis=0)
            tsne_separation = np.linalg.norm(tsne_normal_center - tsne_anomaly_center)
            
            print(f"\nClass Separation:")
            print(f"PCA class center distance: {pca_separation:.4f}")
            print(f"t-SNE class center distance: {tsne_separation:.4f}")
            
            # 判断哪种方法更好
            if pca_separation < 0.1:
                print("⚠️  PCA: Classes are too close - poor linear separability")
            if tsne_separation > pca_separation * 2:
                print("✅ t-SNE: Better class separation than PCA")
        
        print(f"{'='*40}")


def analyze_dimensionality_reduction(model, test_windows, test_labels, model_name: str = "TranAD_QC"):
    """便捷函数：分析PCA和t-SNE降维效果"""
    visualizer = ClassificationVisualizerFixed(save_dir="plots/dimension_analysis")
    visualizer.compare_pca_tsne(model, test_windows, test_labels, model_name)


if __name__ == "__main__":
    print("Fixed Classification Visualization Module")
    print("Usage:")
    print("from visualization_fixed import analyze_dimensionality_reduction")
    print("analyze_dimensionality_reduction(model, test_windows, test_labels, 'your_model')")


