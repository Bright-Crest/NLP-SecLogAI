import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import logging
import time
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from pathlib import Path
import sys
import platform
import matplotlib as mpl
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 中文字体支持配置
def configure_chinese_font():
    """配置中文字体支持"""
    system = platform.system()
    
    # 首先检查字体缓存目录是否存在
    font_cache_dir = os.path.join(os.path.expanduser("~"), ".matplotlib", "fontlist-v330.json")
    if not os.path.exists(os.path.dirname(font_cache_dir)):
        os.makedirs(os.path.dirname(font_cache_dir), exist_ok=True)
    
    try:
        # 根据不同操作系统设置字体
        if system == "Windows":
            # Windows系统
            font_list = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
            for font in font_list:
                try:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    # 测试中文
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.set_title('测试')
                    plt.close(fig)
                    logger.info(f"使用中文字体: {font}")
                    break
                except:
                    continue
        elif system == "Linux":
            # Linux系统
            font_list = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK JP']
            for font in font_list:
                try:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    # 测试中文
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.set_title('测试')
                    plt.close(fig)
                    logger.info(f"使用中文字体: {font}")
                    break
                except:
                    continue
        elif system == "Darwin":
            # MacOS系统
            font_list = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
            for font in font_list:
                try:
                    plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                    # 测试中文
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.set_title('测试')
                    plt.close(fig)
                    logger.info(f"使用中文字体: {font}")
                    break
                except:
                    continue
        
        # 通用设置
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        logger.info("中文字体配置完成")
    except Exception as e:
        logger.warning(f"配置中文字体失败: {e}，将使用系统默认字体")
        # 配置失败时使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']

# 在初始化时配置中文字体
configure_chinese_font()

# 创建自定义配色方案
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_cmap', 
                                              ['#4575b4', '#91bfdb', '#e0f3f8', 
                                               '#ffffbf', '#fee090', '#fc8d59', '#d73027'])

def plot_score_distribution(scores, output_dir=None, model_name="model", ax=None, save_fig=True):
    """
    绘制异常分数分布直方图和KDE曲线
    
    参数:
        scores: 异常分数列表
        output_dir: 输出目录
        model_name: 模型名称
        ax: matplotlib轴对象（可选）
        save_fig: 是否保存图像
    
    返回:
        fig: matplotlib图像对象
        分布统计信息字典
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
        
    # 绘制KDE曲线和直方图
    sns.histplot(scores, kde=True, ax=ax, color="skyblue", stat="density", 
                 bins=30, alpha=0.6, edgecolor="white", linewidth=0.5)
    
    # 使用seaborn的KDE曲线
    sns.kdeplot(scores, ax=ax, color="#d73027", linewidth=2)
    
    # 添加垂直线表示均值和标准差范围
    mean_val = np.mean(scores)
    std_val = np.std(scores)
    median_val = np.median(scores)
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=1, 
               label=f'均值: {mean_val:.4f}')
    ax.axvline(mean_val + std_val, color='green', linestyle=':', linewidth=1, 
               label=f'均值+标准差: {mean_val + std_val:.4f}')
    ax.axvline(median_val, color='blue', linestyle='-.', linewidth=1, 
               label=f'中位数: {median_val:.4f}')
    
    # 添加顶部1%和5%的阈值线
    percentile_95 = np.percentile(scores, 95)
    percentile_99 = np.percentile(scores, 99)
    ax.axvline(percentile_95, color='purple', linestyle=':', linewidth=1, 
               label=f'95%分位数: {percentile_95:.4f}')
    ax.axvline(percentile_99, color='black', linestyle=':', linewidth=1, 
               label=f'99%分位数: {percentile_99:.4f}')
    
    # 计算分数的熵
    # 将分数归一化为概率分布
    hist, bin_edges = np.histogram(scores, bins=30, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    probs = hist * bin_width
    # 计算香农熵
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # 设置中文标题和标签
    ax.set_title(f'异常分数分布 - {model_name}\n熵: {entropy:.4f}', fontsize=14)
    ax.set_xlabel('异常分数', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_fig and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"{model_name}_score_distribution.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"分数分布图已保存到: {fig_path}")
    
    # 计算附加统计信息
    skewness = stats.skew(scores)
    kurtosis = stats.kurtosis(scores)
    iqr = np.percentile(scores, 75) - np.percentile(scores, 25)
    
    stats_info = {
        'mean': float(mean_val),
        'median': float(median_val),
        'std': float(std_val),
        'min': float(np.min(scores)),
        'max': float(np.max(scores)),
        'range': float(np.max(scores) - np.min(scores)),
        'percentile_95': float(percentile_95),
        'percentile_99': float(percentile_99),
        'skewness': float(skewness),  # 衡量分布偏斜度
        'kurtosis': float(kurtosis),  # 衡量分布尖峰度
        'iqr': float(iqr),  # 四分位距
        'entropy': float(entropy),  # 分布熵
        'separation_index': float((np.max(scores) - mean_val) / std_val)  # 最大分数与均值的标准差倍数
    }
    
    return fig, stats_info


def plot_tsne_visualization(features, labels=None, output_dir=None, model_name="model", scores=None, method='tsne', save_fig=True):
    """
    使用t-SNE或UMAP进行特征可视化
    
    参数:
        features: 特征矩阵
        labels: 样本标签（可选）
        output_dir: 输出目录
        model_name: 模型名称
        scores: 异常分数（用于颜色映射）
        method: 降维方法 ('tsne' 或 'umap')
        save_fig: 是否保存图像
    
    返回:
        fig: matplotlib图像对象
    """
    # 确保特征是numpy数组
    features = np.array(features)
    if scores is not None:
        scores = np.array(scores)
    
    # 如果特征过多，进行随机抽样（最多5000个点）
    if len(features) > 5000:
        logger.info(f"特征样本量过大 ({len(features)})，随机抽样5000个点用于可视化")
        indices = np.random.choice(len(features), 5000, replace=False)
        features = features[indices]
        if labels is not None:
            labels = np.array(labels)[indices]
        if scores is not None:
            scores = scores[indices]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    try:
        # 选择降维方法
        if method.lower() == 'tsne':
            logger.info(f"使用t-SNE进行降维 (n_samples={len(features)})...")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            embedding = reducer.fit_transform(features)
        elif method.lower() == 'umap':
            logger.info(f"使用UMAP进行降维 (n_samples={len(features)})...")
            reducer = umap.UMAP(n_components=2, random_state=42, 
                               min_dist=0.1, n_neighbors=min(15, len(features)-1))
            embedding = reducer.fit_transform(features)
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        # 绘制散点图
        if labels is not None:
            # 如果有标签，用不同颜色表示不同类别
            unique_labels = np.unique(labels)
            if len(unique_labels) <= 10:  # 如果标签数量适合离散色彩映射
                cmap = plt.cm.get_cmap('tab10', len(unique_labels))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[cmap(i)], label=f"类别 {label}",
                              alpha=0.7, edgecolors='w', linewidth=0.5)
                ax.legend()
            else:  # 如果标签太多，使用连续色彩映射
                scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                                    c=labels, cmap='viridis',
                                    alpha=0.7, edgecolors='w', linewidth=0.5)
                plt.colorbar(scatter, ax=ax, label='标签值')
        elif scores is not None:
            # 如果有异常分数，用颜色表示分数大小
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                                c=scores, cmap=CUSTOM_CMAP, 
                                alpha=0.7, edgecolors='w', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('异常分数', fontsize=12)
            
            # 添加前10个异常点的标记
            if len(scores) > 10:
                top_indices = np.argsort(scores)[-10:]
                ax.scatter(embedding[top_indices, 0], embedding[top_indices, 1], 
                          s=100, facecolors='none', edgecolors='red', linewidth=2,
                          label='Top-10异常')
                
                # 为顶部异常添加索引标签
                for i, idx in enumerate(top_indices):
                    ax.annotate(f"{i+1}", 
                               (embedding[idx, 0], embedding[idx, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, color='darkred',
                               bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
        else:
            # 如果既没有标签也没有分数，只显示散点图
            ax.scatter(embedding[:, 0], embedding[:, 1], 
                      alpha=0.7, edgecolors='w', linewidth=0.5)
        
        # 设置标题和标签
        ax.set_title(f'特征可视化 ({method.upper()}) - {model_name}', fontsize=14)
        ax.set_xlabel('维度 1', fontsize=12)
        ax.set_ylabel('维度 2', fontsize=12)
        ax.grid(alpha=0.3)
        
        # 如果有标签，计算分离度指标
        separation_metrics = {}
        if labels is not None and len(np.unique(labels)) > 1:
            try:
                # 计算不同类别之间的分离程度
                from sklearn.metrics import silhouette_score
                try:
                    silhouette = silhouette_score(features, labels)
                    separation_metrics['silhouette_score'] = float(silhouette)
                except Exception as e:
                    logger.warning(f"轮廓系数计算失败: {e}")
                
                # 计算类内和类间距离比
                unique_labels = np.unique(labels)
                if len(unique_labels) == 2:  # 二分类情况
                    class0 = features[labels == unique_labels[0]]
                    class1 = features[labels == unique_labels[1]]
                    
                    if len(class0) > 0 and len(class1) > 0:
                        # 计算类内距离（每个类别内样本对的平均距离）
                        from sklearn.metrics.pairwise import euclidean_distances
                        
                        if len(class0) > 1:
                            class0_dist = euclidean_distances(class0).mean()
                        else:
                            class0_dist = 0
                            
                        if len(class1) > 1:
                            class1_dist = euclidean_distances(class1).mean()
                        else:
                            class1_dist = 0
                            
                        intra_dist = (class0_dist + class1_dist) / 2
                        
                        # 计算类间距离（类中心之间的距离）
                        center0 = class0.mean(axis=0)
                        center1 = class1.mean(axis=0)
                        inter_dist = np.linalg.norm(center0 - center1)
                        
                        # 计算分离度（类间距离/类内距离）
                        if intra_dist > 0:
                            separation_index = inter_dist / intra_dist
                            separation_metrics['separation_index'] = float(separation_index)
                            
                            # 在图上添加类中心
                            # 修复：TSNE没有transform方法，不能直接变换类中心
                            # 计算每个类别嵌入点的中心作为类中心
                            class0_embed_points = embedding[labels == unique_labels[0]]
                            class1_embed_points = embedding[labels == unique_labels[1]]
                            
                            if len(class0_embed_points) > 0:
                                center0_embed = np.mean(class0_embed_points, axis=0)
                                ax.scatter(center0_embed[0], center0_embed[1], marker='*', 
                                        s=300, color='blue', edgecolors='black', linewidth=1.5,
                                        label='正常中心')
                            
                            if len(class1_embed_points) > 0:
                                center1_embed = np.mean(class1_embed_points, axis=0)
                                ax.scatter(center1_embed[0], center1_embed[1], marker='*', 
                                        s=300, color='red', edgecolors='black', linewidth=1.5,
                                        label='异常中心')
                            
                            # 标注分离度
                            ax.annotate(f"分离度: {separation_index:.2f}", 
                                      xy=(0.05, 0.95), xycoords='axes fraction',
                                      fontsize=12, color='black',
                                      bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
            except Exception as e:
                logger.warning(f"分离度指标计算失败: {e}")
            
            # 更新图例
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        # 保存图表
        if save_fig and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            fig_path = os.path.join(output_dir, f"{model_name}_{method}_visualization.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征可视化图已保存到: {fig_path}")
        
        return fig, separation_metrics
        
    except Exception as e:
        logger.error(f"特征可视化失败: {e}")
        if fig is not None:
            plt.close(fig)
        raise e


def analyze_spatial_density(embeddings, scores):
    """
    分析二维嵌入空间中的密度关系
    
    参数:
        embeddings: 降维后的二维embeddings
        scores: 每个点的异常分数
    
    返回:
        density_info: 密度分析结果
    """
    # 使用LOF评估局部密度
    try:
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=False)
        lof_scores = -lof.fit_predict(embeddings)
        
        # 计算LOF分数与异常分数的相关性
        correlation = np.corrcoef(lof_scores, scores)[0, 1]
        
        # 按分数排序并分析Top 10%和剩余样本的密度差异
        threshold_idx = int(len(scores) * 0.9)
        sorted_indices = np.argsort(scores)
        
        top_indices = sorted_indices[threshold_idx:]
        rest_indices = sorted_indices[:threshold_idx]
        
        top_density = np.mean(lof_scores[top_indices])
        rest_density = np.mean(lof_scores[rest_indices])
        density_ratio = top_density / (rest_density + 1e-10)
        
        density_info = {
            'lof_correlation': float(correlation),
            'top10pct_density': float(top_density),
            'rest_density': float(rest_density),
            'density_ratio': float(density_ratio)
        }
    except Exception as e:
        logger.warning(f"密度分析失败: {e}")
        density_info = {'error': str(e)}
    
    return density_info


def plot_score_trend_analysis(scores, timestamps=None, threshold=None, window_size=20, output_dir=None, model_name="model", save_fig=True):
    """
    绘制异常分数的时间趋势图和自相关分析
    
    参数:
        scores: 异常分数列表
        timestamps: 时间戳列表（可选）
        threshold: 异常阈值（可选）
        window_size: 滑动窗口大小
        output_dir: 输出目录
        model_name: 模型名称
        save_fig: 是否保存图像
    
    返回:
        fig: matplotlib图像对象
        trend_info: 趋势分析信息字典
    """
    if timestamps is None:
        # 创建简单的序列索引
        timestamps = np.arange(len(scores))
    else:
        # 确保timestamps是numpy数组且长度与scores一致
        timestamps = np.array(timestamps)
        if len(timestamps) != len(scores):
            raise ValueError(f"时间戳长度 ({len(timestamps)}) 与分数长度 ({len(scores)}) 不匹配")
    
    # 确保scores是numpy数组
    scores = np.array(scores)
    
    # 创建图形
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. 时间序列图（上图）
    ax1 = axs[0]
    
    # 绘制分数时间序列
    ax1.plot(timestamps, scores, 'o-', markersize=4, alpha=0.6, color='royalblue', label='异常分数')
    
    # 计算移动平均线
    if len(scores) > window_size:
        window = np.ones(window_size) / window_size
        scores_ma = np.convolve(scores, window, mode='valid')
        timestamps_ma = timestamps[window_size-1:]
        ax1.plot(timestamps_ma, scores_ma, '-', color='red', linewidth=2, label=f'{window_size}点移动平均')
    
    # 如果提供了阈值，绘制阈值线
    if threshold is not None:
        ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'阈值 ({threshold:.4f})')
        
        # 标记超过阈值的点
        anomaly_mask = scores >= threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        if len(anomaly_indices) > 0:
            ax1.scatter(timestamps[anomaly_mask], scores[anomaly_mask], color='red', s=50, 
                       marker='x', label=f'可能异常 ({len(anomaly_indices)}个)')
    
    # 尝试拟合趋势线
    try:
        if len(scores) > 5:  # 至少需要几个点才能拟合
            z = np.polyfit(range(len(scores)), scores, 1)
            p = np.poly1d(z)
            trend_line = p(range(len(scores)))
            ax1.plot(timestamps, trend_line, 'g--', linewidth=1.5, 
                    label=f'趋势线 (斜率={z[0]:.6f})')
            
            # 保存趋势信息
            trend_info = {'trend_slope': float(z[0]), 'trend_intercept': float(z[1])}
        else:
            trend_info = {'trend_slope': 0, 'trend_intercept': 0}
    except Exception as e:
        logger.warning(f"趋势线拟合失败: {e}")
        trend_info = {'trend_slope': 0, 'trend_intercept': 0}
    
    # 设置图表标题和标签
    ax1.set_title(f'异常分数时间趋势 - {model_name}', fontsize=14)
    ax1.set_xlabel('样本序号' if isinstance(timestamps[0], (int, np.integer)) else '时间', fontsize=12)
    ax1.set_ylabel('异常分数', fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')
    
    # 2. 自相关分析（下图）- 检测周期性
    try:
        if len(scores) > 20:  # 需要足够的数据进行自相关分析
            ax2 = axs[1]
            
            # 计算并绘制自相关图
            from statsmodels.tsa.stattools import acf
            lag_max = min(40, len(scores) // 2)  # 最大滞后为数据长度的一半或40，取较小值
            acf_values, confint = acf(scores, nlags=lag_max, alpha=0.05, fft=True)
            
            # 绘制自相关系数
            lags = range(len(acf_values))
            ax2.bar(lags, acf_values, width=0.3, color='royalblue', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # 绘制置信区间
            ax2.fill_between(lags, confint[:, 0], confint[:, 1], color='gray', alpha=0.2)
            
            # 标记显著的自相关
            significant_lags = []
            for i in range(1, len(acf_values)):  # 从1开始，跳过lag=0
                if acf_values[i] > confint[i, 1] or acf_values[i] < confint[i, 0]:
                    significant_lags.append(i)
                    ax2.annotate(f"{i}", xy=(i, acf_values[i]), xytext=(0, 5),
                               textcoords='offset points', ha='center', fontsize=8)
            
            # 寻找最强的自相关
            if len(significant_lags) > 0:
                strongest_lag = max(significant_lags, key=lambda i: abs(acf_values[i]))
                trend_info['has_periodicity'] = True
                trend_info['strongest_autocorrelation_lag'] = strongest_lag
                trend_info['strongest_autocorrelation_value'] = float(acf_values[strongest_lag])
                
                # 标记最强的自相关
                ax2.bar([strongest_lag], [acf_values[strongest_lag]], width=0.3, color='red', alpha=0.9)
                ax2.annotate(f"主要周期: {strongest_lag}", xy=(strongest_lag, acf_values[strongest_lag]), 
                           xytext=(0, 10), textcoords='offset points', ha='center', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
            else:
                trend_info['has_periodicity'] = False
            
            ax2.set_title('自相关分析（检测周期性）', fontsize=12)
            ax2.set_xlabel('滞后', fontsize=10)
            ax2.set_ylabel('自相关系数', fontsize=10)
            ax2.grid(alpha=0.3)
        else:
            # 数据不足，隐藏第二个子图
            axs[1].axis('off')
            trend_info['has_periodicity'] = False
    except Exception as e:
        logger.warning(f"自相关分析失败: {e}")
        axs[1].axis('off')
        trend_info['has_periodicity'] = False
    
    plt.tight_layout()
    
    # 保存图表
    if save_fig and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fig_path = os.path.join(output_dir, f"{model_name}_trend_analysis.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"时间趋势分析图已保存到: {fig_path}")
    
    return fig, trend_info


def generate_topk_samples_report(scores, texts, top_k=10, output_dir=None, model_name="model", other_scores=None, exclude_keywords=None):
    """
    生成Top-K异常样本报告（HTML格式）
    
    参数:
        scores: 异常分数列表
        texts: 文本列表
        top_k: 显示的Top-K样本数量
        output_dir: 输出目录
        model_name: 模型名称
        other_scores: 其他分数字典 {名称: 分数列表}
        exclude_keywords: 排除包含这些关键词的样本
    
    返回:
        html_path: HTML报告文件路径
        top_indices: Top-K样本的索引
    """
    if len(scores) != len(texts):
        raise ValueError(f"分数长度 ({len(scores)}) 与文本长度 ({len(texts)}) 不匹配")
    
    # 转换为numpy数组便于操作
    scores = np.array(scores)
    
    # 获取排序的索引
    sorted_indices = np.argsort(scores)[::-1]  # 从大到小排序
    
    # 如果需要排除某些样本
    if exclude_keywords is not None and len(exclude_keywords) > 0:
        filtered_indices = []
        for idx in sorted_indices:
            text = texts[idx]
            if not any(keyword.lower() in text.lower() for keyword in exclude_keywords):
                filtered_indices.append(idx)
            if len(filtered_indices) >= top_k:
                break
        top_indices = filtered_indices[:top_k]
    else:
        top_indices = sorted_indices[:top_k]
    
    # 提取Top-K样本
    top_scores = scores[top_indices]
    top_texts = [texts[i] for i in top_indices]
    
    # 提取其他分数（如果有）
    other_top_scores = {}
    if other_scores is not None:
        for name, score_list in other_scores.items():
            if len(score_list) == len(scores):
                other_top_scores[name] = np.array(score_list)[top_indices]
    
    # 生成HTML报告
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    html_path = os.path.join(output_dir, f"{model_name}_top{top_k}_anomalies.html")
    
    # 构建HTML内容
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Top-{top_k} 异常样本 - {model_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .summary {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 4px 4px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
                position: sticky;
                top: 0;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .log-text {{
                font-family: monospace;
                white-space: pre-wrap;
                word-wrap: break-word;
                background-color: #f8f9fa;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
                max-height: 200px;
                overflow-y: auto;
            }}
            .score-cell {{
                text-align: center;
                font-weight: bold;
            }}
            .high-score {{
                color: #c0392b;
            }}
            .medium-score {{
                color: #e67e22;
            }}
            .low-score {{
                color: #27ae60;
            }}
            .explanation {{
                font-style: italic;
                color: #7f8c8d;
            }}
            .container {{
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .filters {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                align-items: center;
            }}
            input, select {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            button {{
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 4px;
                cursor: pointer;
            }}
            button:hover {{
                background-color: #2980b9;
            }}
            .hidden {{
                display: none;
            }}
        </style>
    </head>
    <body>
        <h1>Top-{top_k} 异常样本分析</h1>
        
        <div class="summary">
            <p><strong>模型:</strong> {model_name}</p>
            <p><strong>分析时间:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>样本总数:</strong> {len(scores)}</p>
            <p><strong>分数范围:</strong> {np.min(scores):.4f} - {np.max(scores):.4f}</p>
            <p><strong>显示样本数:</strong> {len(top_indices)}</p>
        </div>
        
        <div class="filters">
            <input type="text" id="search-input" placeholder="搜索日志内容...">
            <select id="score-filter">
                <option value="all">所有分数</option>
                <option value="high">高分 (>0.7)</option>
                <option value="medium">中等 (0.4-0.7)</option>
                <option value="low">低分 (<0.4)</option>
            </select>
            <button onclick="applyFilters()">筛选</button>
            <button onclick="resetFilters()">重置</button>
        </div>
        
        <div class="container">
            <table id="anomalies-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>索引</th>
                        <th>异常分数</th>
    """
    
    # 添加其他分数的列标题
    for name in other_top_scores.keys():
        html_content += f"<th>{name}</th>\n"
    
    html_content += """
                        <th>日志内容</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    # 添加每一行数据
    for i, (idx, score, text) in enumerate(zip(top_indices, top_scores, top_texts)):
        # 根据分数确定类别
        if score > 0.7:
            score_class = "high-score"
        elif score > 0.4:
            score_class = "medium-score"
        else:
            score_class = "low-score"
        
        html_content += f"""
                    <tr class="log-row" data-score="{score:.4f}">
                        <td>{i+1}</td>
                        <td>{idx}</td>
                        <td class="score-cell {score_class}">{score:.4f}</td>
        """
        
        # 添加其他分数列
        for name, other_scores_list in other_top_scores.items():
            other_score = other_scores_list[i]
            # 根据其他分数确定类别
            if other_score > 0.7:
                other_score_class = "high-score"
            elif other_score > 0.4:
                other_score_class = "medium-score"
            else:
                other_score_class = "low-score"
            
            html_content += f'<td class="score-cell {other_score_class}">{other_score:.4f}</td>\n'
        
        # 日志内容
        html_content += f"""
                        <td>
                            <div class="log-text">{text.replace("<", "&lt;").replace(">", "&gt;")}</div>
                        </td>
                    </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <script>
            function applyFilters() {
                const searchText = document.getElementById('search-input').value.toLowerCase();
                const scoreFilter = document.getElementById('score-filter').value;
                
                const rows = document.querySelectorAll('tr.log-row');
                
                rows.forEach(row => {
                    const logText = row.querySelector('.log-text').textContent.toLowerCase();
                    const score = parseFloat(row.getAttribute('data-score'));
                    
                    let showByScore = true;
                    if (scoreFilter === 'high') {
                        showByScore = score > 0.7;
                    } else if (scoreFilter === 'medium') {
                        showByScore = score >= 0.4 && score <= 0.7;
                    } else if (scoreFilter === 'low') {
                        showByScore = score < 0.4;
                    }
                    
                    const showByText = searchText === '' || logText.includes(searchText);
                    
                    if (showByScore && showByText) {
                        row.classList.remove('hidden');
                    } else {
                        row.classList.add('hidden');
                    }
                });
            }
            
            function resetFilters() {
                document.getElementById('search-input').value = '';
                document.getElementById('score-filter').value = 'all';
                
                const rows = document.querySelectorAll('tr.log-row');
                rows.forEach(row => {
                    row.classList.remove('hidden');
                });
            }
        </script>
    </body>
    </html>
    """
    
    # 写入HTML文件
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Top-{top_k}异常样本报告已保存到: {html_path}")
    
    return html_path, top_indices


def calculate_consistency_metrics(scores, features=None):
    """
    计算异常分数的内部一致性指标
    
    参数:
        scores: 异常分数列表
        features: 特征矩阵（可选）
        
    返回:
        metrics: 一致性指标字典
    """
    metrics = {}
    
    # 基本统计
    scores_array = np.array(scores)
    mean_val = np.mean(scores_array)
    std_val = np.std(scores_array)
    median_val = np.median(scores_array)
    
    # 1. 分数熵（归一化后）
    try:
        # 使用KDE估计概率密度
        from scipy.stats import gaussian_kde
        
        # 对分数进行归一化，避免数值过小
        min_val = np.min(scores_array)
        range_val = np.max(scores_array) - min_val
        if range_val > 0:
            normalized_scores = (scores_array - min_val) / range_val
        else:
            normalized_scores = np.zeros_like(scores_array)
        
        # 使用高斯核密度估计
        kde = gaussian_kde(normalized_scores)
        
        # 在均匀网格上评估概率密度
        x_grid = np.linspace(0, 1, 100)
        pdf = kde(x_grid)
        pdf = pdf / np.sum(pdf)  # 确保和为1
        
        # 计算熵
        entropy = -np.sum(pdf * np.log2(pdf + 1e-10))
        normalized_entropy = entropy / np.log2(len(x_grid))  # 归一化到[0,1]
        
        metrics['score_entropy'] = float(entropy)
        metrics['normalized_entropy'] = float(normalized_entropy)
    except Exception as e:
        logger.warning(f"分数熵计算失败: {e}")
        metrics['score_entropy_error'] = str(e)
    
    # 2. 分离度指标
    metrics['separation_index'] = float((np.max(scores_array) - mean_val) / (std_val + 1e-10))
    
    # 使用四分位差和极差比率评估异常检测的有效性
    q1 = np.percentile(scores_array, 25)
    q3 = np.percentile(scores_array, 75)
    p99 = np.percentile(scores_array, 99)
    iqr = q3 - q1
    
    # 计算IQR与极差的比率
    if iqr > 0:
        metrics['iqr_range_ratio'] = float(iqr / (np.max(scores_array) - np.min(scores_array) + 1e-10))
    else:
        metrics['iqr_range_ratio'] = 0.0
    
    # 3. 高分段密度指标（高分数样本占比）
    metrics['high_score_density'] = float(np.mean(scores_array > mean_val + 2*std_val))
    metrics['very_high_score_density'] = float(np.mean(scores_array > p99))
    
    # 4. LOF密度分析（如果提供了特征）
    if features is not None and len(features) > 10:  # 至少需要10个样本
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # 使用LOF分析特征空间中的局部离群因子
            n_neighbors = min(20, len(features) // 5)  # 避免邻居数量过大
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.1, n_jobs=-1)
            lof_scores = -lof.fit_predict(features)  # 负值意味着异常
            
            # 计算LOF分数与异常分数的相关性
            correlation = np.corrcoef(lof_scores, scores_array)[0, 1]
            metrics['lof_correlation'] = float(correlation)
            
            # 分析高分数样本的LOF分布
            high_score_mask = scores_array > np.percentile(scores_array, 90)
            if np.any(high_score_mask):
                high_score_lof_mean = np.mean(lof_scores[high_score_mask])
                normal_lof_mean = np.mean(lof_scores[~high_score_mask])
                metrics['high_score_lof_ratio'] = float(high_score_lof_mean / (normal_lof_mean + 1e-10))
            else:
                metrics['high_score_lof_ratio'] = 1.0
        except Exception as e:
            logger.warning(f"LOF分析失败: {e}")
            metrics['lof_analysis_error'] = str(e)
    
    # 5. 相对分布指标（偏度、峰度）
    metrics['skewness'] = float(stats.skew(scores_array))
    metrics['kurtosis'] = float(stats.kurtosis(scores_array))
    
    # 6. 异常检测可靠性指标
    # 基于分数分布的形状评估异常检测质量
    if metrics['skewness'] > 1.0 and metrics['separation_index'] > 2.0:
        metrics['distribution_quality'] = 'good'  # 右偏分布，最大值远离均值
    elif metrics['skewness'] > 0.5 and metrics['separation_index'] > 1.5:
        metrics['distribution_quality'] = 'acceptable'  # 有一定偏度，异常值可辨识
    elif metrics['skewness'] <= 0.5 or metrics['separation_index'] <= 1.0:
        metrics['distribution_quality'] = 'poor'  # 分布接近正态，难以区分异常
    else:
        metrics['distribution_quality'] = 'moderate'
    
    return metrics


def suggest_threshold(scores, method='percentile', percentile=95):
    """
    基于无监督方法推荐异常阈值
    
    参数:
        scores: 异常分数列表
        method: 阈值选择方法，支持'percentile'、'std'、'iqr'
        percentile: 百分位数（当方法为'percentile'时使用）
        
    返回:
        threshold: 推荐的阈值
    """
    scores_array = np.array(scores)
    
    if method == 'percentile':
        # 使用百分位数
        threshold = np.percentile(scores_array, percentile)
    elif method == 'std':
        # 使用均值+n个标准差
        mean_val = np.mean(scores_array)
        std_val = np.std(scores_array)
        threshold = mean_val + 2 * std_val
    elif method == 'iqr':
        # 使用IQR方法（箱线图）
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
    else:
        # 默认使用95%百分位数
        threshold = np.percentile(scores_array, 95)
    
    return float(threshold)


def generate_unsupervised_report(scores, features=None, texts=None, output_dir=None, model_name="model", top_k=20, timestamps=None):
    """
    生成综合的无监督评估报告，包括分数分布、特征可视化、趋势分析和Top-K异常样本报告
    
    参数:
        scores: 异常分数列表
        features: 特征矩阵（可选，用于特征可视化）
        texts: 日志文本列表（可选，用于生成Top-K样本报告）
        output_dir: 输出目录
        model_name: 模型名称
        top_k: Top-K异常样本数量
        timestamps: 时间戳列表（可选，用于趋势分析）
    
    返回:
        报告字典，包含各种评估指标和结果
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "unsupervised_eval")
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"开始生成无监督评估报告，输出目录: {output_dir}")
    
    # 初始化报告字典
    report = {
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_samples": len(scores),
        "threshold": 0.0,  # 将在后面计算
        "num_anomalies": 0,  # 将在后面计算
        "anomaly_ratio": 0.0,  # 将在后面计算
        "suggestions": [],
        "metrics": {
            "distribution": {},
            "consistency": {}
        },
        "assessment": {
            "grade": "未知",
            "score": 0,
            "summary": "",
            "factors": []
        }
    }
    
    # 1. 分数分布分析
    logger.info("1. 生成分数分布分析...")
    fig_dist, dist_stats = plot_score_distribution(
        scores, 
        output_dir=output_dir, 
        model_name=model_name
    )
    
    # 更新报告的分布统计信息
    report["metrics"]["distribution"] = dist_stats
    
    # 计算推荐阈值（使用95%分位数）
    threshold = dist_stats["percentile_95"]
    report["threshold"] = threshold
    
    # 计算异常样本数量
    anomalies = np.where(np.array(scores) >= threshold)[0]
    num_anomalies = len(anomalies)
    report["num_anomalies"] = num_anomalies
    report["anomaly_ratio"] = num_anomalies / len(scores)
    
    # 2. 特征可视化（如果提供了特征）
    if features is not None and len(features) > 0:
        logger.info("2. 生成特征空间可视化...")
        try:
            # 使用异常标签（基于阈值）
            labels = np.zeros(len(scores))
            labels[scores >= threshold] = 1
            
            fig_tsne = plot_tsne_visualization(
                features, 
                labels, 
                output_dir=output_dir, 
                model_name=model_name,
                scores=scores
            )
            
            # 检查特征维度和异常样本的聚集情况
            # TODO: 添加聚集度和分离度指标
            report["metrics"]["consistency"]["features_provided"] = True
        except Exception as e:
            logger.warning(f"特征可视化失败: {e}")
            report["metrics"]["consistency"]["features_provided"] = False
    else:
        logger.info("跳过特征可视化（未提供特征数据）")
        report["metrics"]["consistency"]["features_provided"] = False
    
    # 3. 趋势分析（如果提供了时间戳）
    if timestamps is not None and len(timestamps) == len(scores):
        logger.info("3. 生成时间趋势分析...")
        try:
            fig_trend = plot_score_trend_analysis(
                scores, 
                timestamps, 
                threshold, 
                output_dir=output_dir, 
                model_name=model_name
            )
            report["metrics"]["consistency"]["timestamps_provided"] = True
        except Exception as e:
            logger.warning(f"趋势分析失败: {e}")
            report["metrics"]["consistency"]["timestamps_provided"] = False
    else:
        logger.info("跳过时间趋势分析（未提供时间戳数据）")
        report["metrics"]["consistency"]["timestamps_provided"] = False
    
    # 4. Top-K异常样本报告（如果提供了文本）
    if texts is not None and len(texts) == len(scores):
        logger.info(f"4. 生成Top-{top_k}异常样本报告...")
        try:
            topk_report = generate_topk_samples_report(
                scores, 
                texts, 
                top_k=top_k, 
                output_dir=output_dir, 
                model_name=model_name
            )
            report["metrics"]["consistency"]["texts_provided"] = True
        except Exception as e:
            logger.warning(f"Top-K样本报告生成失败: {e}")
            report["metrics"]["consistency"]["texts_provided"] = False
    else:
        logger.info("跳过Top-K异常样本报告（未提供文本数据）")
        report["metrics"]["consistency"]["texts_provided"] = False
    
    # 5. 内部一致性指标分析
    logger.info("5. 计算内部一致性指标...")
    
    # 分数熵（已在分布分析中计算）
    score_entropy = dist_stats.get("entropy", 0)
    report["metrics"]["consistency"]["score_entropy"] = score_entropy
    
    # 计算分数的变异系数（CV）
    cv = dist_stats.get("std", 0) / (dist_stats.get("mean", 1) + 1e-10)
    report["metrics"]["consistency"]["coefficient_of_variation"] = cv
    
    # 计算局部异常因子（如果有特征）
    if features is not None and len(features) > 0:
        try:
            if len(features) > 10000:  # 如果样本太多，随机抽样计算LOF
                indices = np.random.choice(len(features), 10000, replace=False)
                sampled_features = features[indices]
                lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
                lof_scores = -lof.fit_predict(sampled_features)  # 负值表示异常
                lof_avg = np.mean(lof_scores)
                report["metrics"]["consistency"]["lof_score"] = lof_avg
            else:
                lof = LocalOutlierFactor(n_neighbors=20, novelty=False)
                lof_scores = -lof.fit_predict(features)  # 负值表示异常
                lof_avg = np.mean(lof_scores)
                report["metrics"]["consistency"]["lof_score"] = lof_avg
        except Exception as e:
            logger.warning(f"LOF计算失败: {e}")
            report["metrics"]["consistency"]["lof_score"] = None
    
    # 6. 综合评估
    logger.info("6. 生成综合评估结论...")
    
    # 基于熵、分布和其他指标的综合评估
    assessment_points = 0
    assessment_factors = []
    
    # 评估分数分布
    skewness = dist_stats.get("skewness", 0)
    if skewness > 1.0:
        assessment_points += 20
        assessment_factors.append("分数分布呈现明显的右偏趋势，有利于区分异常")
    elif skewness > 0.5:
        assessment_points += 15
        assessment_factors.append("分数分布呈现右偏趋势，对异常检测有一定帮助")
    else:
        assessment_points += 5
        assessment_factors.append("分数分布接近对称，可能难以区分正常和异常样本")
    
    # 评估熵
    if score_entropy < 2.0:
        assessment_points += 20
        assessment_factors.append("分数熵较低，表明分数分布较为集中，有明确的正常模式")
    elif score_entropy < 3.0:
        assessment_points += 10
        assessment_factors.append("分数熵适中，分数分布有一定的模式")
    else:
        assessment_points += 0
        assessment_factors.append("分数熵较高，分数分布较为混乱，难以区分正常和异常")
    
    # 评估异常比例
    anomaly_ratio = report["anomaly_ratio"]
    if 0.01 <= anomaly_ratio <= 0.05:
        assessment_points += 20
        assessment_factors.append(f"异常比例为{anomaly_ratio:.2%}，在合理范围内")
    elif anomaly_ratio < 0.01:
        assessment_points += 10
        assessment_factors.append(f"异常比例为{anomaly_ratio:.2%}，较低，可能需要降低阈值")
        report["suggestions"].append("考虑降低异常阈值，当前阈值可能过高")
    elif anomaly_ratio <= 0.1:
        assessment_points += 10
        assessment_factors.append(f"异常比例为{anomaly_ratio:.2%}，略高，可能需要提高阈值")
        report["suggestions"].append("考虑提高异常阈值，当前阈值可能过低")
    else:
        assessment_points += 0
        assessment_factors.append(f"异常比例为{anomaly_ratio:.2%}，过高，检测效果可能不理想")
        report["suggestions"].append("异常比例过高，建议重新调整模型或检测方法")
    
    # 评估变异系数
    if cv > 0.5:
        assessment_points += 20
        assessment_factors.append("分数变异系数高，表明模型对不同样本有很好的区分能力")
    elif cv > 0.3:
        assessment_points += 10
        assessment_factors.append("分数变异系数适中，模型有一定的区分能力")
    else:
        assessment_points += 0
        assessment_factors.append("分数变异系数低，模型对不同样本的区分能力有限")
        report["suggestions"].append("模型区分能力有限，建议尝试其他特征或检测方法")
    
    # 评估LOF得分（如果有）
    if "lof_score" in report["metrics"]["consistency"] and report["metrics"]["consistency"]["lof_score"] is not None:
        lof_score = report["metrics"]["consistency"]["lof_score"]
        if lof_score > 1.2:
            assessment_points += 20
            assessment_factors.append("LOF得分高，表明异常样本在特征空间中有明显的离群特性")
        else:
            assessment_points += 10
            assessment_factors.append("LOF得分一般，异常样本在特征空间中的离群特性不太明显")
    
    # 计算最终得分（满分100）
    final_score = min(assessment_points, 100)
    
    # 确定评估等级
    if final_score >= 80:
        grade = "优秀"
        summary = "模型表现优秀，能够有效区分正常和异常样本"
    elif final_score >= 60:
        grade = "良好"
        summary = "模型表现良好，对大部分异常有一定的检测能力"
    elif final_score >= 40:
        grade = "一般"
        summary = "模型表现一般，检测效果有限，可能需要调整"
        report["suggestions"].append("建议尝试其他检测方法或调整特征提取")
    else:
        grade = "不佳"
        summary = "模型表现不佳，检测效果较差，建议重新调整"
        report["suggestions"].append("建议重新训练模型或使用不同的检测方法")
    
    # 更新报告
    report["assessment"]["grade"] = grade
    report["assessment"]["score"] = final_score
    report["assessment"]["summary"] = summary
    report["assessment"]["factors"] = assessment_factors
    
    # 如果没有提供足够的数据，添加相应建议
    if not report["metrics"]["consistency"].get("features_provided", False):
        report["suggestions"].append("建议提供特征数据以进行更全面的评估")
    
    if not report["metrics"]["consistency"].get("timestamps_provided", False):
        report["suggestions"].append("建议提供时间戳数据以进行时间趋势分析")
    
    if not report["metrics"]["consistency"].get("texts_provided", False):
        report["suggestions"].append("建议提供日志文本以生成Top-K异常样本报告")
    
    # 保存报告为JSON文件
    report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    logger.info(f"综合评估报告已保存到: {report_path}")
    
    # 生成Markdown格式的报告
    md_report_path = os.path.join(output_dir, f"{model_name}_evaluation_report.md")
    generate_markdown_report(report, output_dir, md_report_path)
    
    return report

def generate_markdown_report(report, output_dir, output_file=None):
    """
    生成Markdown格式的评估报告
    
    参数:
        report: 评估报告字典
        output_dir: 图表文件所在的输出目录
        output_file: 输出文件路径
    """
    if output_file is None:
        output_file = os.path.join(output_dir, f"{report['model_name']}_evaluation_report.md")
    
    markdown = f"""# 无监督模型评估报告

## 基本信息

- **模型名称**: {report["model_name"]}
- **评估时间**: {report["timestamp"]}
- **样本数量**: {report["num_samples"]}
- **推荐阈值**: {report["threshold"]:.4f}
- **异常样本数**: {report["num_anomalies"]} ({report["anomaly_ratio"]:.2%})

## 模型评估

- **评估等级**: {report["assessment"]["grade"]}
- **评分**: {report["assessment"]["score"]}
- **评估摘要**: {report["assessment"]["summary"]}

### 评估因素

"""
    
    # 添加评估因素
    for factor in report["assessment"]["factors"]:
        markdown += f"- {factor}\n"
    
    # 添加建议
    if report["suggestions"]:
        markdown += "\n## 改进建议\n\n"
        for suggestion in report["suggestions"]:
            markdown += f"- {suggestion}\n"
    
    # 添加分布统计信息
    markdown += "\n## 分数分布统计\n\n"
    markdown += "| 指标 | 值 |\n"
    markdown += "| --- | --- |\n"
    
    dist_stats = report["metrics"]["distribution"]
    for key, value in dist_stats.items():
        if isinstance(value, float):
            markdown += f"| {key} | {value:.4f} |\n"
        else:
            markdown += f"| {key} | {value} |\n"
    
    # 添加一致性指标
    markdown += "\n## 内部一致性指标\n\n"
    markdown += "| 指标 | 值 |\n"
    markdown += "| --- | --- |\n"
    
    consistency = report["metrics"]["consistency"]
    for key, value in consistency.items():
        if isinstance(value, float):
            markdown += f"| {key} | {value:.4f} |\n"
        elif value is None:
            markdown += f"| {key} | 无数据 |\n"
        else:
            markdown += f"| {key} | {value} |\n"
    
    # 添加图表（假设图表已经保存在output_dir中）
    markdown += "\n## 可视化分析\n\n"
    
    # 添加分数分布图
    dist_img = os.path.join(output_dir, f"{report['model_name']}_score_distribution.png")
    if os.path.exists(dist_img):
        rel_path = os.path.basename(dist_img)
        markdown += f"### 分数分布\n\n"
        markdown += f"![分数分布图]({rel_path})\n\n"
    
    # 添加t-SNE图
    tsne_img = os.path.join(output_dir, f"{report['model_name']}_tsne_visualization.png")
    if os.path.exists(tsne_img):
        rel_path = os.path.basename(tsne_img)
        markdown += f"### 特征可视化\n\n"
        markdown += f"![t-SNE特征可视化]({rel_path})\n\n"
    
    # 添加趋势分析图
    trend_img = os.path.join(output_dir, f"{report['model_name']}_trend_analysis.png")
    if os.path.exists(trend_img):
        rel_path = os.path.basename(trend_img)
        markdown += f"### 时间趋势分析\n\n"
        markdown += f"![时间趋势分析]({rel_path})\n\n"
    
    # 添加Top-K异常样本链接
    topk_html = os.path.join(output_dir, f"{report['model_name']}_top{report.get('top_k', 10)}_anomalies.html")
    if os.path.exists(topk_html):
        rel_path = os.path.basename(topk_html)
        markdown += f"### Top异常样本\n\n"
        markdown += f"[查看Top-{report.get('top_k', 10)}异常样本报告]({rel_path})\n\n"
    
    # 保存Markdown报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    logger.info(f"Markdown评估报告已保存到: {output_file}")
    
    return output_file 