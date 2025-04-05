import torch
import numpy as np
import os
import sys
import logging
from sklearn.neighbors import NearestNeighbors

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.models.tinylogbert import create_tiny_log_bert
from app.models.log_tokenizer import LogTokenizer
from app.models.log_window import LogWindow

THRESHOLD = os.environ.get("THRESHOLD", 0.5)

class AnomalyScoreService:
    """
    日志异常评分服务，提供基于预训练模型的异常检测功能
    支持单条日志评分和批量日志评分
    """
    
    def __init__(self, model_dir=None, window_size=10, tokenizer_name='prajjwal1/bert-mini'):
        """
        初始化异常评分服务
        
        参数:
            model_dir: 预训练模型路径
            window_size: 窗口大小
            tokenizer_name: 使用的tokenizer名称
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_dir)
        self.window_size = window_size
        self.log_window = LogWindow(tokenizer_name=tokenizer_name, window_size=window_size)
        self.embeddings_bank = None  # 存储正常日志的embeddings用于KNN分析
        self.knn_model = None
        self.threshold = THRESHOLD # 默认异常阈值
        
        logging.info(f"异常评分服务初始化完成，使用设备: {self.device}")
    
    def _load_model(self, model_dir):
        """加载预训练模型"""
        if model_dir and os.path.isdir(model_dir):
            logging.info(f"从 {os.path.abspath(model_dir)} 加载模型")
            model = create_tiny_log_bert(model_dir)
        else:
            logging.warning(f"模型路径 {os.path.abspath(model_dir)} 不存在，使用未训练的模型")
            model = create_tiny_log_bert()

        model.to(self.device)
        model.eval()
        return model
    
    def score_single_log(self, log_text):
        """
        对单条日志进行异常评分
        
        参数:
            log_text: 日志文本
            
        返回:
            score: 异常分数 (0-1之间，越大越异常)
        """
        # 将单条日志转换为token
        log_tokens = self.log_window.log_tokenizer.tokenize(log_text)
        
        # 将token转移到设备上
        input_ids = log_tokens['input_ids'].to(self.device)
        attention_mask = log_tokens['attention_mask'].to(self.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            score = outputs['anomaly_score'].item()
            
        return score
    
    def score_log_sequence(self, log_list, window_type='fixed', stride=1):
        """
        对日志序列进行异常评分
        
        参数:
            log_list: 日志文本列表
            window_type: 窗口类型 ('fixed' 或 'sliding')
            stride: 滑动窗口的步长
            
        返回:
            scores: 每个窗口的异常分数列表
            avg_score: 平均异常分数
            max_score: 最大异常分数
        """
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
            for i in range(batch['input_ids'].size(0)):
                input_ids = batch['input_ids'][i:i+1]
                attention_mask = batch['attention_mask'][i:i+1]
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                score = outputs['anomaly_score'].item()
                scores.append(score)
                
                # 存储CLS embedding用于KNN分析
                embeddings.append(outputs['cls_embedding'].cpu().numpy())
        
        # 计算KNN分数（如果已有嵌入库）
        if self.knn_model is not None:
            knn_scores = self._compute_knn_scores(embeddings)
            # 融合两种分数（简单平均）
            combined_scores = [(s + k) / 2 for s, k in zip(scores, knn_scores)]
            scores = combined_scores
            
        avg_score = np.mean(scores) if scores else 0.0
        max_score = np.max(scores) if scores else 0.0
        
        return scores, avg_score, max_score
    
    def is_anomaly(self, score):
        """判断分数是否为异常"""
        return score > self.threshold
    
    def build_embedding_bank(self, normal_log_windows):
        """
        构建正常日志的embedding库，用于KNN异常检测
        
        参数:
            normal_log_windows: 正常日志窗口列表
        """
        embeddings = []
        
        for window in normal_log_windows:
            batch = self.log_window.batch_windows([window])
            if batch is None:
                continue
                
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                embeddings.append(outputs['cls_embedding'].cpu().numpy())
        
        if embeddings:
            self.embeddings_bank = np.vstack(embeddings)
            # 构建KNN模型
            self.knn_model = NearestNeighbors(n_neighbors=5).fit(self.embeddings_bank)
            logging.info(f"已构建包含 {len(self.embeddings_bank)} 条正常日志的embedding库")
    
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