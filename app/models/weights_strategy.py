import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VarianceWeightStrategy:
    """
    基于方差的权重调整策略
    
    思想：检测器输出的分数方差越大，代表其区分力越强，权重应越大。
    """
    
    @staticmethod
    def compute_weights(scores, eps=1e-8):
        """
        根据分数的方差计算权重
        
        参数:
            scores: 字典，键为检测器名称，值为检测器分数列表或张量
            eps: 数值稳定性小值，防止除零错误
            
        返回:
            weights: 归一化的权重张量
        """
        # 确保所有分数都是numpy数组
        score_arrays = {}
        for method, score in scores.items():
            if isinstance(score, torch.Tensor):
                score_arrays[method] = score.detach().cpu().numpy()
            elif isinstance(score, list):
                score_arrays[method] = np.array(score)
            else:
                score_arrays[method] = score
        
        # 计算每个检测器的方差
        variances = {}
        for method, score_array in score_arrays.items():
            # 确保分数数组非空
            if len(score_array) > 0:
                # 计算方差
                variances[method] = np.var(score_array)
            else:
                variances[method] = 0.0
        
        # 检查是否所有方差都为零
        if sum(variances.values()) <= eps:
            # 如果所有方差都接近零，使用均匀权重
            uniform_weight = 1.0 / len(variances)
            return {method: uniform_weight for method in variances.keys()}
        
        # 归一化方差作为权重（方差大的检测器权重大）
        total_variance = sum(variances.values()) + eps
        normalized_weights = {method: var / total_variance for method, var in variances.items()}
        
        return normalized_weights
    
    @staticmethod
    def set_ensemble_weights(model, scores):
        """
        设置模型的集成权重
        
        参数:
            model: TinyLogBERT模型实例
            scores: 字典，键为检测器名称，值为检测器分数列表或张量
            
        返回:
            success: 是否成功设置权重
            weights: 设置的权重
        """
        try:
            # 首先检查模型是否支持权重设置
            if not hasattr(model, 'anomaly_ensemble') or not hasattr(model.anomaly_ensemble, 'detector_weights'):
                logger.warning("模型不支持权重设置")
                return False, None
            
            # 确保模型处于静态权重模式
            if model.anomaly_ensemble.fusion_method != 'static_weight':
                # 更改为静态权重模式
                original_fusion_method = model.anomaly_ensemble.fusion_method
                model.anomaly_ensemble.fusion_method = 'static_weight'
                logger.info(f"已将融合方法从 {original_fusion_method} 更改为 static_weight")
            
            # 计算基于方差的权重
            var_weights = VarianceWeightStrategy.compute_weights(scores)
            
            # 根据模型期望的顺序转换权重
            if hasattr(model.anomaly_ensemble, 'detector_weights'):
                weights_tensor = torch.zeros_like(model.anomaly_ensemble.detector_weights)
                
                # 确保检测器顺序匹配
                # 假设顺序为: knn, cluster, lof, iforest, reconstruction
                detector_order = ['knn', 'cluster', 'lof', 'iforest', 'reconstruction']
                
                for i, detector_name in enumerate(detector_order):
                    if detector_name in var_weights and i < len(weights_tensor):
                        weights_tensor[i] = var_weights[detector_name]
                
                # 确保权重和为1
                if weights_tensor.sum() > 0:
                    weights_tensor = weights_tensor / weights_tensor.sum()
                else:
                    # 如果所有权重都是0，则使用均匀权重
                    weights_tensor = torch.ones_like(weights_tensor) / len(weights_tensor)
                
                # 设置模型的融合权重
                model.anomaly_ensemble.detector_weights.data.copy_(weights_tensor)
                logger.info(f"已设置基于方差的融合权重: {weights_tensor.cpu().numpy()}")
                
                return True, weights_tensor.cpu().numpy()
            else:
                logger.warning("模型不支持权重设置")
                return False, None
        except Exception as e:
            logger.error(f"设置权重失败: {e}")
            return False, None
    
    @staticmethod
    def apply_variance_weights(model, score_batches, threshold=None):
        """
        应用方差权重策略并返回调整后的结果
        
        参数:
            model: TinyLogBERT模型实例
            score_batches: 包含多批次分数的列表，每个批次是一个字典
            threshold: 可选的阈值设置，如果为None则使用当前模型的阈值
            
        返回:
            result_dict: 包含原始和调整后统计信息的字典
        """
        # 收集所有批次的分数
        all_method_scores = {}
        
        # 将所有批次的分数合并
        for batch in score_batches:
            if 'method_scores' in batch:
                method_scores = batch['method_scores']
                for method, scores in method_scores.items():
                    if method not in all_method_scores:
                        all_method_scores[method] = []
                    
                    # 确保分数是可迭代的
                    if isinstance(scores, (torch.Tensor, np.ndarray)):
                        scores = scores.tolist()
                    elif not isinstance(scores, list):
                        scores = [scores]
                    
                    all_method_scores[method].extend(scores)
        
        if not all_method_scores:
            logger.warning("未找到有效的检测器分数")
            return {"success": False, "message": "未找到有效的检测器分数"}
        
        # 计算每个方法的方差
        variances = {method: np.var(scores) for method, scores in all_method_scores.items() if len(scores) > 0}
        
        # 应用方差权重策略
        success, weights = VarianceWeightStrategy.set_ensemble_weights(model, all_method_scores)
        
        # 打印权重和方差信息，便于用户了解不同检测器的区分力
        result = {
            "success": success,
            "variances": variances,
            "variance_weights": weights.tolist() if isinstance(weights, np.ndarray) else weights,
            "detector_names": list(all_method_scores.keys()),
            "threshold": threshold
        }
        
        # 给出一些可视化友好的方差解读
        # 方差越大，表示检测器的区分力越强
        variance_interpretations = {}
        if variances:
            max_var = max(variances.values())
            for method, var in variances.items():
                if max_var > 0:
                    normalized_var = var / max_var
                    if normalized_var > 0.8:
                        quality = "极强的区分力"
                    elif normalized_var > 0.6:
                        quality = "很好的区分力"
                    elif normalized_var > 0.4:
                        quality = "中等区分力"
                    elif normalized_var > 0.2:
                        quality = "较弱区分力"
                    else:
                        quality = "很弱区分力"
                    variance_interpretations[method] = quality
            
            result["variance_interpretations"] = variance_interpretations
        
        return result 