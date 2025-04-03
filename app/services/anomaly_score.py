import torch
from transformers import BertModel, BertTokenizer
from app.models.tinylogbert_model import TinyLogBERT
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL_PATH = os.path.join(ROOT_DIR, "ai_detect", "checkpoint", "tiny_model.pt")


class AnomalyScoreService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = 0.5  # 默认阈值，可根据模型实际情况调整

    def load_model(self, model_path=DEFAULT_MODEL_PATH, tokenizer_name="bert-base-uncased"):
        """
        加载预训练的异常检测模型和分词器
        """
        try:
            # 加载BERT-mini基础模型
            base_model = BertModel.from_pretrained('prajjwal1/bert-mini')
            
            # 创建TinyLogBERT模型
            self.model = TinyLogBERT(base_model)
            
            # 如果模型文件存在，加载训练好的权重
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"成功加载模型: {model_path} 到 {self.device}")
            else:
                print(f"警告: 模型文件 {model_path} 不存在，使用未训练的模型")
            
            # 加载分词器
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
            
            # 将模型设置为评估模式
            self.model.to(self.device)
            self.model.eval()
            
            return True
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return False

    def get_anomaly_score(self, log_text):
        """
        对给定的日志文本返回异常分数
        返回: 0到1之间的分数，越高表示异常可能性越大
        """
        if self.model is None or self.tokenizer is None:
            print("错误: 模型未加载")
            return None

        try:
            # 预处理并分词
            tokens = self.tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
            
            # 将tokens移动到正确的设备
            input_ids = tokens['input_ids'].to(self.device)
            attention_mask = tokens['attention_mask'].to(self.device)
            
            # 预测
            with torch.no_grad():
                score = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # 将分数映射到0-1范围内（使用sigmoid）
            score = torch.sigmoid(score).item()
            return float(score)
        except Exception as e:
            print(f"预测异常分数失败: {str(e)}")
            return None

    def is_anomaly(self, log_text, custom_threshold=None):
        """
        判断给定的日志是否为异常
        params:
            log_text: 日志文本
            custom_threshold: 自定义阈值，None则使用默认阈值
        返回: (是否异常, 异常分数)
        """
        threshold = custom_threshold if custom_threshold is not None else self.threshold
        score = self.get_anomaly_score(log_text)
        
        if score is None:
            return False, 0
        
        return score > threshold, score

# 单例服务实例
anomaly_service = AnomalyScoreService()

# 便于导入的函数
def get_anomaly_score(log_text):
    """便于外部调用的函数，获取日志的异常分数"""
    if anomaly_service.model is None:
        anomaly_service.load_model()
    return anomaly_service.get_anomaly_score(log_text)

def is_log_anomaly(log_text, threshold=None):
    """便于外部调用的函数，判断日志是否异常"""
    if anomaly_service.model is None:
        anomaly_service.load_model()
    return anomaly_service.is_anomaly(log_text, threshold) 