import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.manifold import TSNE
import pandas as pd
import os
import json

class EvaluationService:
    """
    评估无盒标日志异常检测效果的服务
    支持ROC-AUC分析、分数分布、t-SNE可视化等
    """
    
    def __init__(self, results_dir="./evaluation_results"):
        """
        初始化评估服务
        params:
            results_dir: 评估结果保存目录
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def evaluate_scores(self, scores, labels, model_name="default"):
        """
        计算ROC-AUC和输出评估结果
        params:
            scores: 模型预测的异常分数列表
            labels: 真实标签列表 (0: 正常, 1: 异常)
            model_name: 模型名称，用于结果文件名
        returns:
            auc: ROC-AUC值
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        # 计算ROC-AUC
        auc = roc_auc_score(labels, scores)
        print(f"ROC-AUC: {auc:.4f}")
        
        # 计算PR-AUC
        ap = average_precision_score(labels, scores)
        print(f"PR-AUC: {ap:.4f}")
        
        # 保存评估结果
        results = {
            "model_name": model_name,
            "roc_auc": float(auc),
            "pr_auc": float(ap),
            "num_samples": len(scores),
            "num_anomalies": int(np.sum(labels)),
            "anomaly_ratio": float(np.sum(labels) / len(labels)),
        }
        
        # 输出评估结果到文件
        output_file = os.path.join(self.results_dir, f"{model_name}_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"评估结果已保存到: {output_file}")
        return auc
    
    def plot_roc_curve(self, scores, labels, model_name="default"):
        """
        绘制ROC曲线并保存
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc = roc_auc_score(labels, scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # 保存ROC曲线
        output_file = os.path.join(self.results_dir, f"{model_name}_roc_curve.png")
        plt.savefig(output_file)
        plt.close()
        print(f"ROC曲线已保存到: {output_file}")
    
    def plot_score_distribution(self, scores, labels, model_name="default"):
        """
        绘制正常和异常样本的分数分布
        """
        scores = np.array(scores)
        labels = np.array(labels)
        
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(normal_scores, bins=50, alpha=0.5, label='正常', color='green')
        plt.hist(anomaly_scores, bins=50, alpha=0.5, label='异常', color='red')
        plt.xlabel('异常分数')
        plt.ylabel('样本数量')
        plt.title('正常与异常样本分数分布')
        plt.legend()
        
        # 保存分布图
        output_file = os.path.join(self.results_dir, f"{model_name}_score_distribution.png")
        plt.savefig(output_file)
        plt.close()
        print(f"分数分布图已保存到: {output_file}")
    
    def plot_tsne_visualization(self, features, labels, model_name="default"):
        """
        使用t-SNE将特征降维并可视化
        params:
            features: 特征矩阵
            labels: 对应的标签
        """
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # 转换为DataFrame便于绘图
        df = pd.DataFrame({
            'x': features_2d[:, 0],
            'y': features_2d[:, 1],
            'label': labels
        })
        
        # 绘制t-SNE图
        plt.figure(figsize=(10, 8))
        for label, color in zip([0, 1], ['green', 'red']):
            subset = df[df['label'] == label]
            plt.scatter(subset['x'], subset['y'], c=color, label='正常' if label == 0 else '异常', alpha=0.6)
        
        plt.title('t-SNE 可视化')
        plt.legend()
        
        # 保存t-SNE图
        output_file = os.path.join(self.results_dir, f"{model_name}_tsne.png")
        plt.savefig(output_file)
        plt.close()
        print(f"t-SNE可视化图已保存到: {output_file}")
    
    def find_optimal_threshold(self, scores, labels):
        """
        找到最佳阈值
        返回: 最佳阈值和对应的F1分数
        """
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # 计算F1分数
        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        # 找到最大F1分数对应的索引
        optimal_idx = np.argmax(f1_scores)
        # 获取对应的阈值
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0
        
        return optimal_threshold, f1_scores[optimal_idx]

# 便于外部调用的函数
evaluation_service = EvaluationService()

def evaluate_scores(scores, labels, model_name="default"):
    """评估异常检测性能"""
    return evaluation_service.evaluate_scores(scores, labels, model_name)

def visualize_results(scores, labels, features=None, model_name="default"):
    """可视化评估结果"""
    evaluation_service.plot_roc_curve(scores, labels, model_name)
    evaluation_service.plot_score_distribution(scores, labels, model_name)
    if features is not None:
        evaluation_service.plot_tsne_visualization(features, labels, model_name)

def find_best_threshold(scores, labels):
    """找到最佳分类阈值"""
    return evaluation_service.find_optimal_threshold(scores, labels) 