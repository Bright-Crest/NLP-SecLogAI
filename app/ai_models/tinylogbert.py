import torch
import os
import sys
import json
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertForMaskedLM
import copy
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.models.weights_strategy import VarianceWeightStrategy


class DynamicLossWeighting(nn.Module):
    """
    动态损失权重调整模块
    支持两种权重调整算法：
    1. GradNorm: 基于梯度范数的权重调整（Chen et al., 2018）
    2. Uncertainty: 基于不确定性的权重调整（Kendall et al., 2018）
    """
    
    def __init__(self, num_losses=3, method='uncertainty', alpha=1.0):
        """
        初始化动态权重模块
        
        参数:
            num_losses: 损失函数的数量
            method: 权重调整方法，'gradnorm'或'uncertainty'
            alpha: GradNorm中的平衡参数，控制任务间平衡速度
        """
        super().__init__()
        self.method = method.lower()
        self.num_losses = num_losses
        self.alpha = alpha
        
        if self.method == 'uncertainty':
            # 对于uncertainty方法，我们学习log(sigma^2)以确保稳定性和正值
            self.log_vars = nn.Parameter(torch.zeros(num_losses))
        elif self.method == 'gradnorm':
            # 对于gradnorm方法，我们直接学习权重
            self.weights = nn.Parameter(torch.ones(num_losses))
            # 保存初始损失值，用于计算相对损失
            self.initial_losses = None
            # 梯度范数历史，用于调试
            self.grad_norms_history = []
        else:
            raise ValueError(f"不支持的权重调整方法: {method}，支持的方法: ['uncertainty', 'gradnorm']")
    
    def forward(self, losses, shared_parameters=None, step=None, backward=False):
        """
        计算加权损失
        
        参数:
            losses: 损失值列表或字典，如 [mlm_loss, contrastive_loss, recon_loss]
            shared_parameters: 用于GradNorm的共享参数
            step: 当前训练步骤 (仅GradNorm需要)
            backward: 是否立即调用backward (仅GradNorm需要)
            
        返回:
            weighted_loss: 加权后的总损失
            weights: 当前各任务权重
        """
        if isinstance(losses, dict):
            losses = list(losses.values())
        
        if len(losses) != self.num_losses:
            raise ValueError(f"提供的损失数量 ({len(losses)}) 与初始化的数量 ({self.num_losses}) 不匹配")
        
        # 确保损失是Tensor
        losses = [loss if isinstance(loss, torch.Tensor) else torch.tensor(loss, device=self.device) 
                 for loss in losses]
        
        if self.method == 'uncertainty':
            return self._uncertainty_weighting(losses)
        elif self.method == 'gradnorm':
            return self._gradnorm_weighting(losses, shared_parameters, step, backward)
    
    def _uncertainty_weighting(self, losses):
        """基于不确定性的权重调整"""
        # 计算权重 = 1 / (2 * sigma^2)
        weights = torch.exp(-self.log_vars)
        
        # 加权损失 = weight * loss + log(sigma^2)
        weighted_losses = [weights[i] * losses[i] + 0.5 * self.log_vars[i] for i in range(self.num_losses)]
        weighted_loss = sum(weighted_losses)
        
        # 返回总损失和当前权重
        return weighted_loss, weights.detach()
    
    def _gradnorm_weighting(self, losses, shared_parameters, step, backward):
        """基于梯度范数的权重调整"""
        if shared_parameters is None:
            raise ValueError("GradNorm方法需要提供共享参数")
        
        if self.initial_losses is None:
            # 首次调用，保存初始损失
            with torch.no_grad():
                self.initial_losses = [loss.item() for loss in losses]
        
        # 计算当前加权损失
        weighted_losses = [self.weights[i] * losses[i] for i in range(self.num_losses)]
        weighted_loss = sum(weighted_losses)
        
        if backward and step is not None and step > 0:
            # 清除之前的梯度
            for param in shared_parameters:
                if param.grad is not None:
                    param.grad.zero_()
            
            # 分别计算每个任务对共享参数的梯度
            grad_norms = []
            for i in range(self.num_losses):
                weighted_losses[i].backward(retain_graph=True)
                
                # 计算梯度L2范数
                grad_norm = 0
                for param in shared_parameters:
                    if param.grad is not None:
                        grad_norm += param.grad.norm(2)**2
                grad_norm = grad_norm.sqrt()
                grad_norms.append(grad_norm)
                
                # 清除梯度，准备下一个任务
                for param in shared_parameters:
                    if param.grad is not None:
                        param.grad.zero_()
            
            # 将梯度范数转换为tensor
            grad_norms = torch.stack(grad_norms)
            self.grad_norms_history.append(grad_norms.detach().cpu().numpy())
            
            # 计算相对损失比例
            with torch.no_grad():
                current_losses = torch.tensor([loss.item() for loss in losses], device=losses[0].device)
                initial_losses = torch.tensor(self.initial_losses, device=losses[0].device)
                loss_ratios = current_losses / initial_losses
                avg_loss_ratio = torch.mean(loss_ratios)
                target_grad_norms = grad_norms.mean() * (loss_ratios / avg_loss_ratio) ** self.alpha
            
            # 计算GradNorm损失，更新权重
            grad_norm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))
            self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]
            
            # 重新计算加权损失，因为权重会更新
            weighted_losses = [self.weights[i] * losses[i] for i in range(self.num_losses)]
            weighted_loss = sum(weighted_losses)
            
            # 由于我们已经手动处理了梯度，不再需要weighted_loss.backward()
            return weighted_loss, self.weights.detach()
        
        # 如果不需要反向传播或是第一步，只返回加权损失
        return weighted_loss, self.weights.detach()
    
    @property
    def device(self):
        """获取权重参数所在设备"""
        if self.method == 'uncertainty':
            return self.log_vars.device
        else:
            return self.weights.device
    
    def get_weights(self):
        """获取当前权重"""
        if self.method == 'uncertainty':
            return torch.exp(-self.log_vars).detach()
        else:
            return self.weights.detach()
    
    def get_loss_weights(self):
        """获取可读性更好的权重字典"""
        weights = self.get_weights()
        # 归一化权重，使总和为num_losses
        normalized_weights = weights * (self.num_losses / weights.sum())
        
        if self.method == 'uncertainty':
            # 对于uncertainty方法，同时返回sigma^2值
            sigmas = torch.exp(self.log_vars).detach()
            return {
                'weights': normalized_weights.cpu().numpy(),
                'sigmas': sigmas.cpu().numpy()
            }
        else:
            return {
                'weights': normalized_weights.cpu().numpy(),
                'initial_losses': self.initial_losses if self.initial_losses else None,
                'grad_history': self.grad_norms_history
            }


class MomentumEncoder(nn.Module):
    """动量编码器，基于MoCo的设计"""
    
    def __init__(self, encoder, momentum=0.999):
        super().__init__()
        # 初始化动量编码器为原编码器的副本
        self.encoder = copy.deepcopy(encoder)
        self.momentum = momentum
        
        # 冻结动量编码器的参数
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 与原编码器相同的前向传播
        return self.encoder(x)
    
    @torch.no_grad()
    def update(self, encoder):
        """更新动量编码器的参数"""
        for param_q, param_k in zip(encoder.parameters(), self.encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)


class UnsupervisedAnomalyDetector:
    """无监督异常检测器，支持多种检测方法"""
    
    def __init__(self, method='knn', n_neighbors=5, n_clusters=10, 
                 fit_memory_size=10000, feature_dim=256):
        """
        初始化无监督异常检测器
        
        参数:
            method: 检测方法，支持 'knn', 'cluster', 'lof', 'iforest'
            n_neighbors: KNN/LOF 的邻居数
            n_clusters: 聚类中心数量
            fit_memory_size: 拟合用的特征记忆库大小
            feature_dim: 特征维度
        """
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_clusters = n_clusters
        
        # 初始化检测器
        if method == 'knn':
            self.detector = NearestNeighbors(n_neighbors=n_neighbors)
        elif method == 'cluster':
            self.detector = KMeans(n_clusters=n_clusters)
        elif method == 'lof':
            self.detector = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
        elif method == 'iforest':
            self.detector = IsolationForest(contamination='auto')
        else:
            raise ValueError(f"不支持的检测方法: {method}")
        
        # 初始化特征记忆库
        self.memory_features = None
        self.fit_memory_size = fit_memory_size
        self.feature_dim = feature_dim
        self.is_fitted = False
        
        # 添加统计数据，用于基于历史数据的归一化
        self.score_stats = {
            'min': None,
            'max': None,
            'sum': 0,
            'count': 0
        }
    
    def update_memory(self, features):
        """更新特征记忆库"""
        features_np = features.detach().cpu().numpy()
        
        if self.memory_features is None:
            self.memory_features = features_np
        else:
            # 如果内存超出限制，随机替换旧特征
            if len(self.memory_features) >= self.fit_memory_size:
                indices_to_replace = np.random.choice(
                    len(self.memory_features), 
                    min(len(features_np), len(self.memory_features)),
                    replace=False
                )
                self.memory_features[indices_to_replace] = features_np[:len(indices_to_replace)]
            else:
                # 直接追加
                self.memory_features = np.vstack([self.memory_features, features_np])
    
    def fit(self, force=False):
        """使用记忆库拟合检测器"""
        if self.memory_features is None or len(self.memory_features) < 10:
            return False  # 内存中样本太少，不进行拟合
        
        # 如果已经拟合过，且内存未变化太多，可以不重新拟合
        if self.is_fitted and not force:
            return True
        
        try:
            if self.method == 'knn':
                self.detector.fit(self.memory_features)
                # 计算记忆库中所有样本的异常分数，用于统计
                distances, _ = self.detector.kneighbors(self.memory_features)
                memory_scores = np.mean(distances, axis=1)
                
            elif self.method == 'cluster':
                self.detector.fit(self.memory_features)
                # 计算记忆库中所有样本的异常分数，用于统计
                distances = self.detector.transform(self.memory_features)
                memory_scores = np.min(distances, axis=1)
                
            elif self.method == 'lof' or self.method == 'iforest':
                self.detector.fit(self.memory_features)
                # 计算记忆库中所有样本的异常分数，用于统计
                if hasattr(self.detector, 'decision_function'):
                    memory_scores = -self.detector.decision_function(self.memory_features)
            
            self.update_score_stats(memory_scores)
            self.is_fitted = True
            return True
        except Exception as e:
            print(f"拟合异常检测器失败: {e}")
            return False
    
    def update_score_stats(self, scores):
        """更新分数统计信息，用于基于历史数据的归一化"""
        if scores is None or len(scores) == 0:
            return
            
        current_min = float(np.min(scores))
        current_max = float(np.max(scores))
        
        # 更新最小值和最大值
        if self.score_stats['min'] is None or current_min < self.score_stats['min']:
            self.score_stats['min'] = current_min
        
        if self.score_stats['max'] is None or current_max > self.score_stats['max']:
            self.score_stats['max'] = current_max
        
        # 更新总和和计数，用于计算均值（可选）
        self.score_stats['sum'] += float(np.sum(scores))
        self.score_stats['count'] += len(scores)
    
    def get_anomaly_score(self, features):
        """
        计算异常分数
        
        参数:
            features: 特征向量，形状为 [batch_size, feature_dim]
            
        返回:
            scores: 异常分数，形状为 [batch_size]，值越大表示越异常
        """
        # 确保特征是numpy格式
        if isinstance(features, torch.Tensor):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = features
        
        # 如果检测器未拟合，返回零分数
        if not self.is_fitted:
            return torch.zeros(len(features_np), device=features.device)
        
        # 根据不同方法计算异常分数
        if self.method == 'knn':
            # KNN距离方法：计算到最近邻居的平均距离
            distances, _ = self.detector.kneighbors(features_np)
            # 使用均值作为异常分数
            scores = np.mean(distances, axis=1)
            
        elif self.method == 'cluster':
            # 聚类距离方法：计算到最近聚类中心的距离
            distances = self.detector.transform(features_np)
            # 使用最小距离作为异常分数
            scores = np.min(distances, axis=1)
            
        elif self.method == 'lof':
            # LOF方法：直接使用LOF的预测分数
            # LOF返回负数，值越小表示越异常，需要取相反数
            scores = -self.detector.decision_function(features_np)
            
        elif self.method == 'iforest':
            # Isolation Forest方法：使用决策函数的负值
            # 返回负数，值越小表示越异常，需要取相反数
            scores = -self.detector.decision_function(features_np)
        
        max_val = np.max(scores)
        min_val = np.min(scores)
        if max_val > min_val:
            scores = (scores - min_val) / (max_val - min_val)
        else:
            scores = np.clip(scores, 0, 1)
        
        # 转换为tensor并返回
        return torch.FloatTensor(scores).to(features.device)


class ReconstructionHead(nn.Module):
    """基于自编码器的重构头，用于计算重构误差作为异常分数"""
    
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        # 编码器：将隐藏状态压缩到瓶颈层
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_size),
        )
        
        # 解码器：从瓶颈层重构原始隐藏状态
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
        )
        
        # 添加统计数据，用于基于历史数据的归一化
        self.register_buffer('error_min', torch.tensor(float('inf')))
        self.register_buffer('error_max', torch.tensor(float('-inf')))
        self.register_buffer('error_sum', torch.tensor(0.0))
        self.register_buffer('error_count', torch.tensor(0))
        
        # 是否已经累积了足够的统计数据
        self.have_stats = False
    
    def update_error_stats(self, errors):
        """更新误差统计信息"""
        if errors is None or len(errors) == 0:
            return
            
        with torch.no_grad():
            batch_min = errors.min().item()
            batch_max = errors.max().item()
            
            # 更新最小值和最大值
            if batch_min < self.error_min:
                self.error_min = torch.tensor(batch_min, device=self.error_min.device)
            
            if batch_max > self.error_max:
                self.error_max = torch.tensor(batch_max, device=self.error_max.device)
            
            # 更新总和和计数
            self.error_sum += errors.sum().item()
            self.error_count += len(errors)
            
            # 检查是否已累积足够的统计数据
            if self.error_count > 100 and (self.error_max - self.error_min) > 1e-6:
                self.have_stats = True
    
    def forward(self, hidden_states, update_stats=True):
        """计算重构误差"""
        # 只使用[CLS]向量
        cls_output = hidden_states[:, 0]
        
        # 编码-解码过程
        bottleneck = self.encoder(cls_output)
        reconstructed = self.decoder(bottleneck)
        
        # 计算重构误差（MSE）
        reconstruction_error = F.mse_loss(reconstructed, cls_output, reduction='none')
        # 沿特征维度求和，得到每个样本的重构误差
        sample_errors = reconstruction_error.sum(dim=1)
        
        # 更新统计信息
        if not self.have_stats or update_stats:
            self.update_error_stats(sample_errors)
        
        return sample_errors, bottleneck
    
    def normalize_error(self, sample_errors):
        """归一化误差"""
        # 基于历史统计数据进行归一化，而不是基于当前批次
        if self.have_stats:
            min_val, max_val = self.error_min.item(), self.error_max.item()
            if max_val > min_val:
                normalized_errors = (sample_errors - min_val) / (max_val - min_val)
                # 对于超出历史范围的异常值进行处理
                normalized_errors = torch.clamp(normalized_errors, 0, 1)
                return normalized_errors
        else:
            # 如果没有足够的统计数据，则使用批次归一化作为备选
            if len(sample_errors) > 1:
                min_val, max_val = sample_errors.min(), sample_errors.max()
                if max_val > min_val:
                    normalized_errors = (sample_errors - min_val) / (max_val - min_val)
            normalized_errors = torch.clamp(normalized_errors, 0, 1)
            return normalized_errors
        
    def get_stats(self):
        """获取误差统计信息"""
        return {
            'min': self.error_min.item(),
            'max': self.error_max.item(),
            'mean': (self.error_sum / max(1, self.error_count)).item(),
            'count': self.error_count.item(),
            'have_stats': self.have_stats
        }


class AnomalyScoreEnsemble(nn.Module):
    """
    异常分数融合器，将多个异常检测器的结果融合为一个更稳健的总分数
    实现"专家融合"机制，每个检测方法视为一个"专家评判"
    """
    
    def __init__(self, num_detectors=4, fusion_method='dynamic_weight'):
        """
        初始化融合器
        
        参数:
            num_detectors: 检测器数量
            fusion_method: 融合方法，可选 'dynamic_weight', 'static_weight', 'max', 'mean'
        """
        super().__init__()
        self.num_detectors = num_detectors
        self.fusion_method = fusion_method
        
        # 初始化各检测器权重
        if fusion_method == 'dynamic_weight':
            self.detector_weights = nn.Parameter(torch.ones(num_detectors) * 1/num_detectors)
        elif fusion_method == 'static_weight':
            # 固定权重，但可手动设置
            self.register_buffer('detector_weights', torch.ones(num_detectors) * 1/num_detectors)
        
        # 记录检测器性能指标
        self.detector_performance = {}
        self.update_count = 0
    
    def forward(self, detector_scores):
        """
        融合多个检测器的分数
        
        参数:
            detector_scores: 字典或列表，包含各检测器的异常分数
            
        返回:
            ensemble_score: 融合后的异常分数
            weights: 各检测器的权重
        """
        # 将输入转换为统一格式
        if isinstance(detector_scores, dict):
            scores = list(detector_scores.values())
            detector_names = list(detector_scores.keys())
        else:
            scores = detector_scores
            detector_names = [f"detector_{i}" for i in range(len(scores))]
        
        # 确保分数数量与初始化时指定的一致
        if len(scores) != self.num_detectors:
            raise ValueError(f"检测器分数数量({len(scores)})与初始化时指定的数量({self.num_detectors})不一致")
        
        # 将分数转换为tensor并确保在同一设备上
        scores = [s if isinstance(s, torch.Tensor) else torch.tensor(s, device=self.device) 
                 for s in scores]
        
        # 根据不同融合方法计算最终分数
        if self.fusion_method == 'max':
            # 取最大值作为最终分数
            ensemble_score = torch.max(torch.stack(scores), dim=0)[0]
            # max方法没有实际权重，但为了统一返回格式，生成虚拟权重
            weights = torch.zeros(self.num_detectors, device=self.device)
            max_indices = torch.argmax(torch.stack(scores), dim=0)
            for i in range(self.num_detectors):
                weights[i] = (max_indices == i).float().mean()
            
        elif self.fusion_method == 'mean':
            # 简单平均
            ensemble_score = torch.mean(torch.stack(scores), dim=0)
            weights = torch.ones(self.num_detectors, device=self.device) / self.num_detectors
            
        elif self.fusion_method == 'dynamic_weight':
            # 使用基于方差的权重策略计算动态权重
            detector_scores_dict = {detector_names[i]: scores[i] for i in range(len(scores))}
            var_weights = VarianceWeightStrategy.compute_weights(detector_scores_dict)
        
            # 创建权重向量
            weights = torch.zeros(self.num_detectors, device=self.device)
            for i, name in enumerate(detector_names):
                if name in var_weights:
                    weights[i] = torch.tensor(var_weights[name], device=self.device)
            
            weights = torch.ones(self.num_detectors, device=self.device) * 1/self.num_detectors
            # 确保权重和为1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # 如果所有权重都是0，则使用均匀权重
                weights = torch.ones_like(weights) / self.num_detectors
        
            # 更新模型的动态权重参数
            with torch.no_grad():
                self.detector_weights.copy_(weights)  # 使用log将权重转换为logits
            
            # 加权求和计算最终分数
            # 将分数堆叠成矩阵，然后进行矩阵乘法计算加权和
            stacked_scores = torch.stack(scores, dim=0)  # [num_detectors, batch_size]
            ensemble_score = torch.matmul(self.detector_weights, stacked_scores)
            
        elif self.fusion_method == 'static_weight':
            # 使用softmax获取权重，确保权重和为1且非负
            norm_weights = F.softmax(self.detector_weights, dim=0)
            
            # 加权求和
            # 将分数堆叠成矩阵，然后进行矩阵乘法计算加权和
            stacked_scores = torch.stack(scores, dim=0)  # [num_detectors, batch_size]
            ensemble_score = torch.matmul(norm_weights, stacked_scores)
                
            weights = norm_weights
        
        # 记录本次融合的信息
        self.update_count += 1
        for i, name in enumerate(detector_names):
            if name not in self.detector_performance:
                self.detector_performance[name] = {
                    'weight_history': [],
                    'score_stats': {'min': [], 'max': [], 'mean': [], 'std': []}
                }
            
            # 记录权重历史
            self.detector_performance[name]['weight_history'].append(weights[i].item())
            
            # 记录分数统计信息
            with torch.no_grad():
                score = scores[i]
                if len(score) > 0:
                    self.detector_performance[name]['score_stats']['min'].append(score.min().item())
                    self.detector_performance[name]['score_stats']['max'].append(score.max().item())
                    self.detector_performance[name]['score_stats']['mean'].append(score.mean().item())
                    self.detector_performance[name]['score_stats']['std'].append(score.std(unbiased=False).item())
        
        return ensemble_score, weights
    
    def set_weights(self, weights):
        """手动设置静态权重"""
        if self.fusion_method != 'static_weight':
            raise ValueError(f"只有static_weight融合方法支持手动设置权重，当前方法: {self.fusion_method}")
        
        if len(weights) != self.num_detectors:
            raise ValueError(f"权重数量({len(weights)})与检测器数量({self.num_detectors})不一致")
        
        # 更新权重
        if isinstance(weights, torch.Tensor):
            self.detector_weights.copy_(weights)
        else:
            self.detector_weights.copy_(torch.tensor(weights, device=self.detector_weights.device))
    
    def get_performance_summary(self):
        """获取检测器性能汇总"""
        summary = {
            'update_count': self.update_count,
            'detector_weights': {},
            'score_stats': {}
        }
        
        for name, perf in self.detector_performance.items():
            if perf['weight_history']:
                # 计算平均权重
                avg_weight = sum(perf['weight_history']) / len(perf['weight_history'])
                summary['detector_weights'][name] = avg_weight
            
            # 计算分数统计汇总
            score_summary = {}
            for stat_name, stat_values in perf['score_stats'].items():
                if stat_values:
                    score_summary[stat_name] = sum(stat_values) / len(stat_values)
            
            summary['score_stats'][name] = score_summary
        
        return summary
    
    @property
    def device(self):
        """获取权重参数所在设备"""
        return self.detector_weights.device


class TinyLogBERT(BertForMaskedLM):
    """
    基于BERT-mini的日志异常检测模型，同时支持MLM损失和异常分数输出
    增强版：添加MoCo风格的动量编码器和队列，用于更稳定的对比学习
    无监督版：移除AnomalyScoreHead，使用无监督方法检测异常
    
    该模型继承自BertForMaskedLM，因此原生支持掩码语言建模任务
    """
    
    def __init__(self, config):
        super().__init__(config)
        # bert和mlm_head已在父类BertForMaskedLM中初始化
        
        # 重构头，用于计算重构误差
        self.reconstruction_head = ReconstructionHead(config.hidden_size)
        
        # 无监督异常检测器
        self.anomaly_methods = {
            'knn': UnsupervisedAnomalyDetector(method='knn', n_neighbors=5, feature_dim=config.hidden_size),
            'cluster': UnsupervisedAnomalyDetector(method='cluster', n_clusters=10, feature_dim=config.hidden_size),
            'lof': UnsupervisedAnomalyDetector(method='lof', n_neighbors=20, feature_dim=config.hidden_size),
            'iforest': UnsupervisedAnomalyDetector(method='iforest', feature_dim=config.hidden_size)
        }
        self.current_method = 'knn'  # 默认使用KNN检测方法
        
        # 动量编码器（key encoder）
        self.momentum_bert = MomentumEncoder(self.bert)
        
        # 初始化可训练温度参数
        self.temperature = nn.Parameter(torch.tensor(0.1))
        
        # 特征队列
        self.queue_size = 4096  # 队列大小
        self.register_buffer("queue", torch.randn(self.queue_size, config.hidden_size))
        self.queue = F.normalize(self.queue, dim=1)  # 初始化为单位向量
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 添加动态损失权重模块
        self.loss_weighting = DynamicLossWeighting(num_losses=3, method='uncertainty')
        
        # 异常分数融合器 - 支持基于方差的权重策略
        self.anomaly_ensemble = AnomalyScoreEnsemble(
            num_detectors=5,  # KNN, Cluster, LOF, IForest, Reconstruction
            fusion_method='dynamic_weight'
        )
        
        # 是否启用融合模式
        self.enable_ensemble = True
        
        # 记录累积的方法分数，用于方差权重计算
        self.accumulated_method_scores = {}
        self.score_accumulation_limit = 1000  # 限制累积分数数量，避免内存过大
        
        self.init_weights()
    
    def set_detection_method(self, method):
        """设置异常检测方法"""
        if method == 'ensemble':
            self.enable_ensemble = True
            return
            
        self.enable_ensemble = False
        if method not in self.anomaly_methods and method != 'reconstruction':
            raise ValueError(f"不支持的检测方法: {method}，可选值: {list(self.anomaly_methods.keys()) + ['reconstruction', 'ensemble']}")
        self.current_method = method
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列"""
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 将keys添加到队列中
        if batch_size <= self.queue_size:
            if ptr + batch_size > self.queue_size:
                # 队列已满，需要重新从头开始
                self.queue[:(ptr + batch_size) % self.queue_size] = keys[:(self.queue_size - ptr)]
                self.queue[ptr:] = keys[:(self.queue_size - ptr)]
                self.queue[:(ptr + batch_size) % self.queue_size] = keys[(self.queue_size - ptr):]
            else:
                self.queue[ptr:ptr + batch_size] = keys
        
        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    @staticmethod
    def get_augmented_log(input_text, tokenizer):
        """对日志文本进行简单数据增强"""
        
        # 获取词汇，以便替换
        words = input_text.split()
        if len(words) <= 3:  # 太短的文本不做增强
            return input_text
        
        # 随机选择增强策略
        strategy = random.choice(['delete', 'shuffle', 'replace'])
        
        if strategy == 'delete':
            # 随机删除一些词汇（最多删除30%）
            max_to_delete = max(1, int(len(words) * 0.3))
            num_to_delete = random.randint(1, max_to_delete)
            indices_to_delete = sorted(random.sample(range(len(words)), num_to_delete), reverse=True)
            for idx in indices_to_delete:
                words.pop(idx)
            
        elif strategy == 'shuffle':
            # 随机打乱部分词汇的顺序
            if len(words) >= 4:
                shuffle_range = random.randint(2, min(len(words), 5))
                start_idx = random.randint(0, len(words) - shuffle_range)
                segment = words[start_idx:start_idx + shuffle_range]
                random.shuffle(segment)
                words[start_idx:start_idx + shuffle_range] = segment
        
        elif strategy == 'replace':
            # 随机替换一些词汇
            max_to_replace = max(1, int(len(words) * 0.2))
            num_to_replace = random.randint(1, max_to_replace)
            
            for _ in range(num_to_replace):
                idx = random.randint(0, len(words) - 1)
                # 使用近似长度的随机字符替换
                word_len = len(words[idx])
                replacement = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=max(2, word_len)))
                words[idx] = replacement
        
        return ' '.join(words)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        positive_pairs=None,  # 用于对比学习的正样本对
        augment_batch=False,  # 是否对批次进行数据增强
        tokenizer=None,       # 用于数据增强的tokenizer
        training_phase=True,  # 是否处于训练阶段
        update_memory=True,   # 是否更新特征内存
    ):
        # 调用父类BertForMaskedLM的forward方法
        mlm_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 总是获取隐藏状态
            return_dict=True,
        )
       
        # 获取MLM的loss
        mlm_loss = mlm_outputs.loss
        
        # 获取序列输出用于异常检测和对比学习
        sequence_output = mlm_outputs.hidden_states[-1]
        
        # 获取query的[CLS]嵌入并归一化
        q = sequence_output[:, 0]  # [batch_size, hidden_size]
        q_norm = F.normalize(q, dim=1)
        
        # 获取设备信息，用于后续张量创建
        device = q.device if q is not None else (
            input_ids.device if input_ids is not None else 
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # 初始化对比损失和对比距离
        contrastive_loss = torch.tensor(0.0).to(device)
        contrastive_distances = torch.zeros(q_norm.size(0)).to(device)  # 对每个样本的对比距离
        
        # 计算重构损失
        reconstruction_error, bottleneck = self.reconstruction_head(sequence_output, update_stats=training_phase)
        
        # MoCo风格的对比学习
        if augment_batch and tokenizer is not None:
            # 数据增强方式生成正样本对
            with torch.no_grad():
                # 1. 更新momentum encoder
                self.momentum_bert.update(self.bert)
                
                # 2. 处理输入生成key features
                # 获取batch中的文本
                texts = []
                for i in range(input_ids.size(0)):
                    # 解码当前样本
                    text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    texts.append(text)
                
                # 数据增强生成正样本对
                augmented_texts = [self.get_augmented_log(text, tokenizer) for text in texts]
                # 对增强后的文本进行编码
                augmented_inputs = tokenizer(augmented_texts, padding='max_length', 
                                           truncation=True, return_tensors='pt',
                                           max_length=input_ids.size(1)).to(input_ids.device)
                
                # 使用momentum encoder处理增强样本
                augmented_outputs = self.momentum_bert.encoder(
                    input_ids=augmented_inputs['input_ids'],
                    attention_mask=augmented_inputs['attention_mask'],
                    return_dict=True,
                    output_hidden_states=True
                )
                if positive_pairs is not None:
                    positive_pairs = augmented_outputs
                
                # 获取并归一化key特征
                k = augmented_outputs.hidden_states[-1][:, 0]  # [batch_size, hidden_size]
                k = F.normalize(k, dim=1)
                
                # 入队操作
                self._dequeue_and_enqueue(k)
        
        elif positive_pairs is not None:
            # 使用提供的正样本对
            # 假设positive_pairs为[batch_size, 2]的形式，表示正样本对索引
            if positive_pairs.size(0) > 0:
                with torch.no_grad():
                    # 更新momentum encoder
                    self.momentum_bert.update(self.bert)
                    
                    # 选择正样本的输入
                    pos_input_ids = torch.index_select(input_ids, 0, positive_pairs[:, 1])
                    pos_attention_mask = torch.index_select(attention_mask, 0, positive_pairs[:, 1])
                    
                    # 使用momentum encoder处理正样本
                    pos_outputs = self.momentum_bert.encoder(
                        input_ids=pos_input_ids,
                        attention_mask=pos_attention_mask,
                        return_dict=True,
                        output_hidden_states=True
                    )
                    
                    # 获取并归一化key特征
                    k = pos_outputs.hidden_states[-1][:, 0]  # [batch_size, hidden_size]
                    k = F.normalize(k, dim=1)
                    
                    # 入队操作
                    self._dequeue_and_enqueue(k)
        
        # 计算对比损失
        if augment_batch or (positive_pairs is not None and positive_pairs.size(0) > 0):
            # 获取可训练的温度参数（限制在合理范围内）
            temperature = torch.clamp(self.temperature, 0.05, 0.5)
            
            # 计算正样本对的相似度
            if augment_batch:
                # 使用增强样本作为正样本
                l_pos = torch.einsum('nc,nc->n', [q_norm, k]).unsqueeze(-1)  # [batch_size, 1]
            else:
                # 使用提供的正样本对
                q_pos = torch.index_select(q_norm, 0, positive_pairs[:, 0])
                l_pos = torch.einsum('nc,nc->n', [q_pos, k]).unsqueeze(-1)  # [batch_size, 1]
            
            # 使用队列作为负样本
            l_neg = torch.einsum('nc,ck->nk', [q_norm, self.queue.t()])  # [batch_size, queue_size]
            
            # InfoNCE损失
            logits = torch.cat([l_pos, l_neg], dim=1)  # [batch_size, 1+queue_size]
            logits = logits / temperature
            
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input_ids.device)  # 正样本的索引为0
            
            contrastive_loss = F.cross_entropy(logits, labels)
            
            # 计算每个样本的对比距离（用于增强异常评分）
            with torch.no_grad():
                # 计算query与队列中所有样本的距离
                distances = 1 - torch.mm(q_norm, self.queue.t())  # [batch_size, queue_size]
                # 获取每个样本的最小距离（与最相似样本的距离）
                min_distances, _ = torch.min(distances, dim=1)  # [batch_size]
                # 对距离进行归一化
                if min_distances.max() > min_distances.min():
                    contrastive_distances = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min())
                else:
                    contrastive_distances = min_distances * 0.0  # 所有距离相同，设为0
        
        # 训练阶段更新特征内存
        if training_phase and update_memory and not self.training:
            # 只在评估模式下更新内存
            with torch.no_grad():
                # 更新特征内存
                for method in self.anomaly_methods.values():
                    method.update_memory(q)
        
        # 使用无监督方法计算异常分数
        with torch.no_grad():
            # 对于每个检测方法，尝试拟合并计算分数
            for name, detector in self.anomaly_methods.items():
                # 如果是当前使用的方法且未拟合，尝试拟合
                if name == self.current_method and not detector.is_fitted:
                    detector.fit()
            
            # 计算各个检测方法的异常分数
            method_scores = {}
            for name, detector in self.anomaly_methods.items():
                method_scores[name] = detector.get_anomaly_score(q)
            
            # 添加重构误差作为一种检测方法
            method_scores['reconstruction'] = self.reconstruction_head.normalize_error(reconstruction_error)
            
            # 累积方法分数，用于计算方差权重
            if self.enable_ensemble and self.anomaly_ensemble.fusion_method == 'dynamic_weight':
                for name, scores in method_scores.items():
                    if name not in self.accumulated_method_scores:
                        self.accumulated_method_scores[name] = []
                    
                    # 将张量转换为列表并添加
                    if isinstance(scores, torch.Tensor):
                        scores_list = scores.detach().cpu().tolist()
                    else:
                        scores_list = list(scores)
                    
                    self.accumulated_method_scores[name].extend(scores_list)
                    
                    # 限制累积分数数量
                    if len(self.accumulated_method_scores[name]) > self.score_accumulation_limit:
                        # 保留最新的数据
                        self.accumulated_method_scores[name] = self.accumulated_method_scores[name][-self.score_accumulation_limit:]
            
            # 决定使用单一方法还是融合多种方法
            if self.enable_ensemble:
                # 使用融合器合并多个检测结果
                if self.anomaly_ensemble.fusion_method == 'dynamic_weight' and len(self.accumulated_method_scores) > 0:
                    # 确保至少已累积了一定数量的分数
                    min_samples_required = 10
                    if all(len(scores) >= min_samples_required for scores in self.accumulated_method_scores.values()):
                        # 使用累积的方法分数计算方差权重
                        anomaly_score, method_weights = self.anomaly_ensemble(method_scores)
                    else:
                        # 样本不足，使用普通方式融合
                        anomaly_score, method_weights = self.anomaly_ensemble(method_scores)
                else:
                    # 使用常规方法融合
                    anomaly_score, method_weights = self.anomaly_ensemble(method_scores)
            else:
                # 使用指定的单一方法
                if self.current_method == 'reconstruction':
                    anomaly_score = reconstruction_error
                else:
                    anomaly_score = method_scores[self.current_method]
                # 创建一个空的方法权重占位符
                method_weights = torch.zeros(len(method_scores), device=q.device)
        
        # 使用动态损失权重计算总损失
        if self.training and labels is not None:
            # 分别计算各个损失
            losses = {
                'mlm': mlm_loss,
                'contrastive': contrastive_loss,
                'reconstruction': reconstruction_error.mean()
            }
            
            # 如果使用GradNorm，我们需要共享参数
            if hasattr(self.loss_weighting, 'method') and self.loss_weighting.method == 'gradnorm':
                # 使用最后一层的参数作为共享参数
                shared_parameters = list(self.bert.encoder.layer[-1].parameters())
                # 获取当前步骤，这里简化处理
                step = 1
                # 计算加权损失
                total_loss, loss_weights = self.loss_weighting(
                    losses=[mlm_loss, contrastive_loss, reconstruction_error.mean()],
                    shared_parameters=shared_parameters,
                    step=step,
                    backward=True
                )
            else:
                # 对于Uncertainty方法，直接计算加权损失
                total_loss, loss_weights = self.loss_weighting(
                    losses=[mlm_loss, contrastive_loss, reconstruction_error.mean()]
                )
            
            # 保存当前权重供返回
            current_weights = self.loss_weighting.get_weights()
        else:
            # 非训练状态，简单相加
            # 检查损失是否为None，如果是则用0替代
            mlm_loss = mlm_loss if mlm_loss is not None else torch.tensor(0.0, device=reconstruction_error.device)
            contrastive_loss = contrastive_loss if isinstance(contrastive_loss, torch.Tensor) else torch.tensor(0.0, device=reconstruction_error.device)
            reconstruction_mean = reconstruction_error.mean() if reconstruction_error is not None else torch.tensor(0.0, device=contrastive_loss.device)
            total_loss = mlm_loss + contrastive_loss + reconstruction_mean
            current_weights = torch.tensor([1.0, 1.0, 1.0], device=total_loss.device)
        
        # 构建返回结果
        return {
            "loss": total_loss,
            "mlm_loss": mlm_loss,
            "contrastive_loss": contrastive_loss,
            "anomaly_score": anomaly_score,
            "contrastive_distances": contrastive_distances,
            "reconstruction_error": reconstruction_error,
            "bottleneck": bottleneck,
            "cls_embedding": q,
            "logits": mlm_outputs.logits,  # MLM的预测结果
            "temperature": self.temperature,  # 当前温度参数
            "loss_weights": current_weights,  # 当前损失权重
            "method_scores": {k: v.detach() for k, v in method_scores.items()},  # 各检测方法分数
            "method_weights": method_weights.detach()  # 各检测方法权重
        }
    
    def get_cls_embedding(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        """仅获取[CLS]向量用于后续分析"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1][:, 0]
    
    def save_pretrained(self, save_directory, save_config=True, **kwargs):
        """
        保存模型权重及异常检测器特征内存
        
        参数:
            save_directory: 保存目录
            save_config: 是否保存配置文件
            **kwargs: 传递给父类save_pretrained的其他参数
        """
        # 调用父类的save_pretrained方法保存模型权重
        super().save_pretrained(save_directory, save_config=save_config, **kwargs)
        
        # 保存异常检测器特征内存
        detector_dir = os.path.join(save_directory, "detectors")
        os.makedirs(detector_dir, exist_ok=True)
        
        # 保存各个检测器的特征内存
        for method, detector in self.anomaly_methods.items():
            if hasattr(detector, "memory_features") and detector.memory_features is not None:
                feature_path = os.path.join(detector_dir, f"{method}_features.npy")
                np.save(feature_path, detector.memory_features)
                
                # 保存检测器参数和统计信息
                params = {
                    "method": detector.method,
                    "n_neighbors": detector.n_neighbors,
                    "n_clusters": detector.n_clusters,
                    "fit_memory_size": detector.fit_memory_size,
                    "feature_dim": detector.feature_dim,
                    "is_fitted": detector.is_fitted,
                    "score_stats": detector.score_stats  # 保存分数统计信息
                }
                param_path = os.path.join(detector_dir, f"{method}_params.json")
                with open(param_path, "w") as f:
                    json.dump(params, f)
        
        # 保存重构头的统计信息
        recon_stats = self.reconstruction_head.get_stats()
        recon_stats_path = os.path.join(detector_dir, "reconstruction_stats.json")
        with open(recon_stats_path, "w") as f:
            json.dump(recon_stats, f)
        
        # 保存累积的方法分数（用于方差权重计算）
        if hasattr(self, "accumulated_method_scores") and self.accumulated_method_scores:
            scores_path = os.path.join(detector_dir, "accumulated_method_scores.json")
            # 确保分数是可序列化的
            serializable_scores = {}
            for method, scores in self.accumulated_method_scores.items():
                # 限制保存的分数数量，避免文件过大
                max_scores_to_save = min(1000, len(scores))
                serializable_scores[method] = scores[-max_scores_to_save:]
            
            with open(scores_path, "w") as f:
                json.dump(serializable_scores, f)
        
        # 保存融合器性能指标
        if hasattr(self.anomaly_ensemble, "detector_performance"):
            ensemble_path = os.path.join(detector_dir, "ensemble_performance.json")
            with open(ensemble_path, "w") as f:
                # 转换为可序列化格式（去除tensor）
                performance = self.anomaly_ensemble.get_performance_summary()
                json.dump(performance, f)
        
        # 保存当前使用的检测方法和融合模式
        config_path = os.path.join(detector_dir, "detector_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "current_method": self.current_method,
                "enable_ensemble": self.enable_ensemble,
                "score_accumulation_limit": getattr(self, "score_accumulation_limit", 1000)
            }, f)
        
        print(f"异常检测器特征内存和配置已保存至 {detector_dir}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        从预训练文件加载模型权重及异常检测器特征内存
        
        参数:
            pretrained_model_name_or_path: 预训练模型路径或名称
            *model_args: 传递给父类from_pretrained的位置参数
            **kwargs: 传递给父类from_pretrained的关键字参数
            
        返回:
            model: 加载了权重和特征内存的模型
        """
        # 调用父类的from_pretrained方法加载模型权重
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 检查是否存在检测器目录
        detector_dir = os.path.join(pretrained_model_name_or_path, "detectors")
        if not os.path.exists(detector_dir):
            print(f"未找到检测器特征目录: {detector_dir}，将使用空特征内存")
            return model
        
        # 加载重构头的统计信息
        recon_stats_path = os.path.join(detector_dir, "reconstruction_stats.json")
        if os.path.exists(recon_stats_path):
            try:
                with open(recon_stats_path, "r") as f:
                    recon_stats = json.load(f)
                
                # 恢复重构头的统计信息
                model.reconstruction_head.error_min.copy_(torch.tensor(recon_stats['min']))
                model.reconstruction_head.error_max.copy_(torch.tensor(recon_stats['max']))
                model.reconstruction_head.error_sum.copy_(torch.tensor(recon_stats['mean'] * recon_stats['count']))
                model.reconstruction_head.error_count.copy_(torch.tensor(recon_stats['count']))
                model.reconstruction_head.have_stats = recon_stats.get('have_stats', False)
                
                print(f"已加载重构头的统计信息")
            except Exception as e:
                print(f"加载重构头统计信息时出错: {e}")
        
        # 加载检测方法配置
        config_path = os.path.join(detector_dir, "detector_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                model.current_method = config.get("current_method", "knn")
                model.enable_ensemble = config.get("enable_ensemble", True)
                model.score_accumulation_limit = config.get("score_accumulation_limit", 1000)
        
        # 加载累积的方法分数
        scores_path = os.path.join(detector_dir, "accumulated_method_scores.json")
        if os.path.exists(scores_path):
            try:
                with open(scores_path, "r") as f:
                    model.accumulated_method_scores = json.load(f)
                print(f"已加载累积方法分数，用于方差权重计算")
            except Exception as e:
                print(f"加载累积方法分数时出错: {e}")
                model.accumulated_method_scores = {}
        else:
            model.accumulated_method_scores = {}
        
        # 加载各个检测器的特征内存
        for method, detector in model.anomaly_methods.items():
            feature_path = os.path.join(detector_dir, f"{method}_features.npy")
            if os.path.exists(feature_path):
                try:
                    # 加载特征内存
                    detector.memory_features = np.load(feature_path)
                    
                    # 加载检测器参数
                    param_path = os.path.join(detector_dir, f"{method}_params.json")
                    if os.path.exists(param_path):
                        with open(param_path, "r") as f:
                            params = json.load(f)
                            for param_name, param_value in params.items():
                                if hasattr(detector, param_name):
                                    setattr(detector, param_name, param_value)
                    
                    # 重新拟合检测器，但保留加载的统计信息
                    if detector.memory_features is not None and len(detector.memory_features) >= 10:
                        # 备份统计信息，防止fit过程覆盖
                        score_stats_backup = detector.score_stats.copy() if hasattr(detector, 'score_stats') else None
                        
                        detector.fit(force=True)
                        
                        # 如果有备份的统计信息且fit过程覆盖了，则恢复
                        if score_stats_backup and detector.score_stats != score_stats_backup:
                            detector.score_stats = score_stats_backup
                            
                        print(f"检测器 {method} 已加载 {len(detector.memory_features)} 个特征样本并重新拟合")
                    else:
                        print(f"检测器 {method} 的特征样本数量不足，跳过拟合")
                
                except Exception as e:
                    print(f"加载检测器 {method} 的特征内存时出错: {e}")
            else:
                print(f"未找到检测器 {method} 的特征内存文件")
        
        print(f"异常检测器特征内存和配置已从 {detector_dir} 加载")
        return model


def create_tiny_log_bert(model_dir=None):
    """创建一个小型的BERT模型用于日志分析"""
    if model_dir is None:
        config = BertConfig.from_pretrained(
            "prajjwal1/bert-mini",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        model = TinyLogBERT.from_pretrained("prajjwal1/bert-mini", config=config)
    else:
        config = BertConfig.from_pretrained(model_dir)
        model = TinyLogBERT.from_pretrained(model_dir, config=config)
    return model 