import os
import sys
import logging
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.ai_models.anomaly_detector import AnomalyDetector
from app.ai_models.log_window import LogWindow

# 从环境变量获取配置
THRESHOLD = float(os.environ.get("THRESHOLD", 0.5))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 128))
USE_KNN = os.environ.get("USE_KNN", "False").lower() in ("true", "1", "yes")


class AnomalyScoreService:
    """
    日志异常评分服务，提供基于预训练模型的异常检测功能
    支持单条日志评分和批量日志评分
    底层使用AnomalyDetector进行实现
    支持KNN增强的异常检测
    """
    
    def __init__(self, model_dir=None, window_size=10, tokenizer_name='prajjwal1/bert-mini', detection_method='ensemble'):
        """
        初始化异常评分服务
        
        参数:
            model_dir: 预训练模型路径
            window_size: 窗口大小
            tokenizer_name: 使用的tokenizer名称
            detection_method: 异常检测方法
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.threshold = THRESHOLD # 默认异常阈值
        self.batch_size = BATCH_SIZE
        self.use_knn = USE_KNN
        
        # 初始化底层的异常检测器
        self.detector = AnomalyDetector(
            model_dir=model_dir,
            window_size=window_size,
            tokenizer_name=tokenizer_name,
            detection_method=detection_method
        )
        
        # 保留log_window实例以保持兼容性
        self.log_window = LogWindow(tokenizer_name=tokenizer_name, window_size=window_size)
        
        # 初始化KNN相关属性以保持兼容性
        self.embeddings_bank = None
        self.knn_model = None
        
        logging.info(f"异常评分服务初始化完成，使用设备: {self.device}, KNN增强: {self.use_knn}")
    
    @property
    def model(self):
        """返回底层模型以保持兼容性"""
        return self.detector.model
    
    def score_single_log(self, log_text, use_knn=None):
        """
        对单条日志进行异常评分
        
        参数:
            log_text: 日志文本
            use_knn: 是否使用KNN增强，None表示使用默认设置
            
        返回:
            score: 异常分数 (0-1之间，越大越异常)
        """
        # 确定是否使用KNN
        should_use_knn = self.use_knn if use_knn is None else use_knn
        
        if not should_use_knn or self.knn_model is None:
            # 不使用KNN或KNN模型未准备好，直接使用detector的detect方法
            result = self.detector.detect(
                log_text=log_text, 
                threshold=self.threshold
            )
            return result['score']
        else:
            # 使用KNN增强
            # 将单条日志转换为token
            log_tokens = self.log_window.log_tokenizer.tokenize(log_text)
            
            # 将token转移到设备上
            input_ids = log_tokens['input_ids'].to(self.device)
            attention_mask = log_tokens['attention_mask'].to(self.device)
            
            # 执行推理获取embedding和基本分数
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                base_score = outputs['anomaly_score'].item()
                cls_embedding = outputs['cls_embedding'].cpu().numpy()
            
            # 计算KNN分数
            knn_scores = self._compute_knn_scores([cls_embedding])
            knn_score = knn_scores[0]
            
            # 融合两种分数（简单平均）
            combined_score = (base_score + knn_score) / 2
            
            return combined_score
    
    def score_log_sequence(self, log_list, window_type='fixed', stride=1, use_knn=None):
        """
        对日志序列进行异常评分
        
        参数:
            log_list: 日志文本列表
            window_type: 窗口类型 ('fixed' 或 'sliding')
            stride: 滑动窗口的步长
            use_knn: 是否使用KNN增强，None表示使用默认设置
            
        返回:
            scores: 每个窗口的异常分数列表
            avg_score: 平均异常分数
            max_score: 最大异常分数
        """
        # 确定是否使用KNN
        should_use_knn = self.use_knn if use_knn is None else use_knn
        
        if not should_use_knn or self.knn_model is None:
            # 不使用KNN或KNN模型未准备好，直接使用detector的detect_sequence方法
            result = self.detector.detect_sequence(
                log_list=log_list,
                window_type=window_type,
                stride=stride,
                threshold=self.threshold,
                batch_size=self.batch_size
            )
            
            # 从结果中提取分数信息
            if not result or 'windows' not in result:
                return [], 0.0, 0.0
            
            scores = [window['score'] for window in result['windows']]
            avg_score = result['avg_score'] if 'avg_score' in result else np.mean(scores) if scores else 0.0
            max_score = result['max_score'] if 'max_score' in result else np.max(scores) if scores else 0.0
            
            return scores, avg_score, max_score
        else:
            # 使用KNN增强
            # 根据窗口类型处理日志序列
            if window_type == 'fixed':
                window_tokens, _ = self.log_window.create_fixed_windows(log_list)
            else:  # sliding
                window_tokens = self.log_window.create_sliding_windows(log_list, stride)
            
            if not window_tokens:
                return [], 0.0, 0.0
            
            # 批量处理窗口
            batch = self.log_window.batch_windows(window_tokens)
            
            if batch is None:
                return [], 0.0, 0.0
            
            # 将batch移到设备上
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 执行推理
            scores = []
            embeddings = []
            
            with torch.no_grad():
                for i in range(0, batch['input_ids'].size(0), self.batch_size):
                    # 处理一个批次
                    end_idx = min(i + self.batch_size, batch['input_ids'].size(0))
                    input_ids = batch['input_ids'][i:end_idx]
                    attention_mask = batch['attention_mask'][i:end_idx]
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_scores = outputs['anomaly_score'].cpu().numpy()
                    scores.extend(batch_scores.tolist())
                    
                    # 存储CLS embedding用于KNN分析
                    batch_embeddings = outputs['cls_embedding'].cpu().numpy()
                    embeddings.extend([batch_embeddings[j] for j in range(batch_embeddings.shape[0])])
            
            # 计算KNN分数
            knn_scores = self._compute_knn_scores(embeddings)
            
            # 融合两种分数（简单平均）
            combined_scores = [(s + k) / 2 for s, k in zip(scores, knn_scores)]
            
            avg_score = np.mean(combined_scores) if combined_scores else 0.0
            max_score = np.max(combined_scores) if combined_scores else 0.0
            
            return combined_scores, avg_score, max_score
    
    def is_anomaly(self, score):
        """判断分数是否为异常"""
        return score > self.threshold
    
    def build_embedding_bank(self, normal_log_windows):
        """
        构建正常日志的embedding库，用于KNN异常检测
        
        参数:
            normal_log_windows: 正常日志窗口列表
        """
        # 使用detector中的方法提取normal_log_windows中的文本
        normal_logs = []
        for window in normal_log_windows:
            if isinstance(window, dict) and 'text' in window:
                normal_logs.append(window['text'])
            elif isinstance(window, str):
                normal_logs.append(window)
        
        if normal_logs:
            # 将detector的detection_method设置为'knn'，然后使用_collect_features_for_detector
            original_method = self.detector.detection_method
            self.detector.detection_method = 'knn'
            
            # 创建数据集并收集特征
            window_texts = self.detector.prepare_log_windows(normal_logs, window_type='fixed')
            
            if window_texts:
                from torch.utils.data import Dataset
                
                class SimpleDataset(Dataset):
                    def __init__(self, texts, tokenizer):
                        self.texts = texts
                        self.tokenizer = tokenizer
                        self.encodings = tokenizer(texts, truncation=True, padding='max_length', return_tensors='pt')
                    
                    def __len__(self):
                        return len(self.texts)
                    
                    def __getitem__(self, idx):
                        return {key: val[idx] for key, val in self.encodings.items()}
                
                dataset = SimpleDataset(window_texts, self.detector.tokenizer)
                self.detector._collect_features_for_detector(dataset, batch_size=self.batch_size)
                
                # 恢复原来的检测方法
                self.detector.detection_method = original_method
                
                # 提取embeddings和KNN模型以保持兼容性
                if 'knn' in self.detector.model.anomaly_methods:
                    knn_detector = self.detector.model.anomaly_methods['knn']
                    if hasattr(knn_detector, 'reference_embeddings') and knn_detector.reference_embeddings is not None:
                        self.embeddings_bank = knn_detector.reference_embeddings
                        self.knn_model = knn_detector.model
                        # 开启KNN增强
                        self.use_knn = True
                        
                        logging.info(f"已构建包含 {len(self.embeddings_bank)} 条正常日志的embedding库，KNN增强已启用")
                        return True
        
        # 如果没有成功构建，返回False
        logging.warning("KNN嵌入库构建失败或没有提供有效的正常日志")
        return False
    
    def _compute_knn_scores(self, embeddings):
        """
        计算KNN异常分数
        
        参数:
            embeddings: 待评估的embedding列表
            
        返回:
            scores: KNN异常分数列表
        """
        if self.knn_model is None:
            return [0.0] * len(embeddings)
            
        embeddings = np.vstack(embeddings)
        distances, _ = self.knn_model.kneighbors(embeddings)
        # 将距离转换为分数（0-1之间）
        mean_distances = np.mean(distances, axis=1)
        max_dist = np.max(mean_distances) if len(mean_distances) > 0 else 1.0
        scores = mean_distances / (max_dist + 1e-10)  # 归一化
        
        return scores
    
    def set_threshold(self, threshold):
        """设置异常阈值"""
        self.threshold = threshold
        logging.info(f"异常阈值已设置为 {threshold}")
    
    def set_use_knn(self, use_knn):
        """设置是否使用KNN增强"""
        self.use_knn = use_knn
        status = "启用" if use_knn else "禁用"
        logging.info(f"KNN增强已{status}")


# 全局实例，便于外部调用
anomaly_service = None

def init_anomaly_service(model_dir=None, window_size=10):
    """初始化异常评分服务"""
    global anomaly_service
    anomaly_service = AnomalyScoreService(model_dir, window_size)
    return anomaly_service

def get_anomaly_service():
    """获取异常评分服务实例"""
    global anomaly_service
    if anomaly_service is None:
        anomaly_service = AnomalyScoreService()
    return anomaly_service 