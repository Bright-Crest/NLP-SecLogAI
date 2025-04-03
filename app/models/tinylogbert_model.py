from transformers import BertModel
import torch.nn as nn
import torch

class TinyLogBERT(nn.Module):
    def __init__(self, base_model=None):
        super().__init__()
        if base_model is None:
            base_model = BertModel.from_pretrained('prajjwal1/bert-mini')
        self.encoder = base_model
        self.anomaly_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 返回异常分
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        return self.anomaly_head(x)
    
    def get_embedding(self, input_ids, attention_mask):
        """
        提取日志嵌入向量（便于可视化或聚类）
        """
        # 进入评估模式
        self.eval()
        
        # 获取编码器输出
        with torch.no_grad():
            embedding = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        
        return embedding 