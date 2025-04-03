# 🚀 TinyLogBERT - 无盒标日志异常检测简单版项目设计

## 目标

在 2 天内完成一个可运行、可扩展、可清晰量化无盒标日志异常检测系统原型，基于 TinyLogBERT 微调。

---

## 内容

### ✅ 模块化设计

| 模块 | 功能 |
|--------|------|
| `log_parser.py` | 日志转换 Token/Text |
| `anomaly_detector.py` | 微调 TinyBERT + 输出异常得分 |
| `anomaly_routes.py` | API接口，接收日志，返回异常结果 |
| `evaluate.py` | ROC-AUC/分数分布等量化无盒标效果 |

---

## 日志数据集

### 选择: [HDFS 日志数据集](https://github.com/logpai/loghub)
- 日志格式简单，含有模板 ID，适合无盒标学习
- 选取 5~10 万条日志进行预处理

```
2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx
```

转换为:
```
["dfsclient", "successfully", "read", "block"]
```

---

## 目录结构扩展

在你的目录基础上，增加:

```
app/
|— models/
|   |— tinylogbert_model.py     # 基于 BERT-mini 的 TinyLogBERT + 异常头
|— services/
|   |— log_tokenizer.py         # 转 token 分词器
|   |— anomaly_score.py         # 根据输入日志返回异常分
|   |— evaluate.py              # 效果量化模块
```

---

## 核心代码

### 【models/tinylogbert_model.py】
```python
from transformers import BertModel
import torch.nn as nn

class TinyLogBERT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.anomaly_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 返回异常分
        )

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        return self.anomaly_head(x)
```

### 【services/anomaly_score.py】
```python
import torch
from transformers import BertTokenizer
from app.models.tinylogbert_model import TinyLogBERT

base_model = BertModel.from_pretrained('prajjwal1/bert-mini')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TinyLogBERT(base_model)
model.load_state_dict(torch.load("./checkpoint/model.pt"))
model.eval()

def get_anomaly_score(log_text):
    tokens = tokenizer(log_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        score = model(**tokens)
    return float(score.item())
```

### 【services/evaluate.py】
```python
from sklearn.metrics import roc_auc_score

def evaluate_scores(scores, labels):
    auc = roc_auc_score(labels, scores)
    print("ROC-AUC:", auc)
    return auc
```

---

## 开发时间表

| 时间 | 任务 |
|------|------|
| D1-AM | 数据转化，加载 BERT-mini，建立 TinyLogBERT + Head |
| D1-PM | 输入日志 转分词，培训 MLM 五轮，保存模型 |
| D2-AM | 运行 score，扩展 Flask API + CLI |
| D2-PM | 量化 evaluate.py 输出 AUC + t-SNE 可视化 |

---

如需要：
- 按需附上 notebook 模型训练脚本
- 提供 FastAPI/分布式支持
- 改造异常头输出 softmax 分级分类

