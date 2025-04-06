# 🔍 TinyLogBERT 日志异常检测系统

基于轻量级BERT模型的无监督日志异常检测系统，支持无标签学习和多种窗口策略。

## 📋 项目概述

本项目使用BERT-mini模型，配合固定窗口和滑动窗口方法构造日志行为序列，实现无需标注的日志异常检测系统。
核心特点包括：

* **轻量级模型**：使用BERT-mini (11M参数) 作为基础，减少资源占用
* **无标签学习**：使用正常日志进行自监督训练，无需异常标注数据
* **多窗口支持**：同时支持固定窗口和滑动窗口两种方式
* **融合评分**：结合模型得分、KNN和自编码器等多种得分方法
* **易于部署**：提供REST API和命令行接口

## 🔢 技术架构

```
+-----------------+        +------------------+        +----------------+
| 日志数据处理模块 |  --->  | TinyLogBERT模型  |  --->  | 异常评分与检测 |
+-----------------+        +------------------+        +----------------+
      |                            |                         |
      v                            v                         v
+-------------+           +--------------+          +----------------+
| 固定/滑动窗口 |           | MLM预训练     |          | API/CLI接口    |
+-------------+           +--------------+          +----------------+
```

## 🚀 主要组件

1. **模型架构** (app/models/tinylogbert.py)
   - BERT-mini编码器 (4层Transformer)
   - 异常评分MLP头 (从CLS token计算异常分数)

2. **数据处理** (ai_detect/log_window.py)
   - 支持固定窗口和滑动窗口
   - 日志文本预处理和标准化

3. **异常检测** (ai_detect/anomaly_detector.py)
   - 模型训练与评估
   - 单条/序列日志异常检测
   - 结果可视化与阈值优化

4. **服务层** (app/services/anomaly_score.py)
   - 异常评分服务
   - KNN嵌入分析
   - 多种评分融合

5. **接口** (app/routes/ai_routes.py & ai_detect/ai_detect_cli.py)
   - RESTful API
   - 命令行工具

## 📥 安装与依赖

需要以下依赖：

```
flask
transformers
torch
numpy
scikit-learn
matplotlib
pandas
tqdm
seaborn
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 🔧 使用方法

### 训练模型

```bash
python ai_detect/train.py --train_file ./path/to/normal_logs.txt \
                         --model_dir ./ai_detect/checkpoint \
                         --window_size 10 \
                         --num_epochs 3
```

### 评估模型

```bash
python ai_detect/evaluate.py --test_file ./path/to/test_logs.json \
                           --model_path ./ai_detect/checkpoint/model.pt
```

### 命令行检测

```bash
# 检测单条日志
python ai_detect/ai_detect_cli.py --logs "failed login attempt from ip" \
                                --model-path ./ai_detect/checkpoint/model.pt

# 检测日志文件
python ai_detect/ai_detect_cli.py --log-file ./path/to/logs.txt \
                                --model-path ./ai_detect/checkpoint/model.pt \
                                --window-type sliding
```

### API使用示例

```python
import requests
import json

# 单条日志检测
response = requests.post('http://localhost:5000/ai/score_log', 
                        json={'log': 'failed login attempt from ip'})
result = response.json()
print(f"异常分数: {result['score']}, 是否异常: {result['is_anomaly']}")

# 日志序列检测
logs = ["login attempt", "password incorrect", "account locked"]
response = requests.post('http://localhost:5000/ai/score_log_sequence', 
                        json={'logs': logs, 'window_type': 'sliding'})
result = response.json()
print(f"平均分数: {result['avg_score']}, 异常窗口数: {result['num_anomaly_windows']}")
```

## 📊 评估指标

系统使用以下指标评估性能：

- **ROC-AUC**：衡量二分类性能
- **PR-AUC**：特别适用于不平衡数据
- **分数分布**：观察正常/异常样本的分数分布
- **t-SNE可视化**：直观展示样本聚类情况

## 🔮 扩展功能

- 支持多模型融合评分
- 自动阈值调整
- 异常日志解释
- 主客体行为路径分析
- 新增GNN模型分析网络行为

## 📜 项目结构

```
NLP-SecLogAI/
├── ai_detect/                # AI检测相关代码
│   ├── anomaly_detector.py   # 异常检测器
│   ├── log_window.py         # 日志窗口处理
│   ├── train.py              # 训练脚本
│   ├── evaluate.py           # 评估脚本
│   ├── ai_detect_cli.py      # 命令行接口
│   ├── checkpoint/           # 模型保存目录
│   └── output/               # 输出结果目录
├── app/
│   ├── models/
│   │   └── tinylogbert.py    # 模型架构
│   ├── services/
│   │   ├── log_tokenizer.py  # 日志标记化
│   │   ├── evaluate.py       # 评估服务
│   │   └── anomaly_score.py  # 异常评分服务
│   └── routes/
│       └── ai_routes.py      # API接口
└── requirements.txt          # 项目依赖
```

## 📝 许可

[MIT License](LICENSE)

## 🙏 致谢

- HuggingFace Transformers 库
- scikit-learn 项目
- LogPAI 数据集 