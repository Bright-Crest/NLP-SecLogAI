from transformers import DistilBertModel
import torch
import torch.nn as nn

class LogBERTDistil(nn.Module):
    """
    LogBERT-Distil模型实现
    使用DistilBERT作为基础模型，添加异常检测头
    相比TinyLogBERT，这个模型更为精简但保持了较好的性能
    """
    def __init__(self, base_model=None):
        super().__init__()
        if base_model is None:
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        else:
            self.encoder = base_model
        
        # DistilBERT的隐藏层大小为768
        self.anomaly_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 返回异常分
        )

    def forward(self, input_ids, attention_mask):
        # DistilBERT输出的最后隐藏状态
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 取[CLS]位置的表示作为整个序列的表示
        x = outputs.last_hidden_state[:, 0, :]
        
        # 通过异常检测头进行预测
        return self.anomaly_head(x)
    
    def get_embedding(self, input_ids, attention_mask):
        """
        提取日志嵌入向量（便于可视化或聚类）
        """
        # 进入评估模式
        self.eval()
        
        # 获取编码器输出
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[:, 0, :]
        
        return embedding 