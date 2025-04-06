# NLP-SecLogAI
> 基于自然语言处理的安全日志分析与异常检测系统

## 项目概述

NLP-SecLogAI 是一个创新型安全日志分析系统，利用先进的自然语言处理技术对安全日志进行智能化处理。本系统突破了传统 SIEM 系统（如 Splunk、ELK）仅依赖规则匹配的局限，实现了对日志的语义理解和智能化处理，有效降低误报率，提高新型攻击的检测能力。

## 核心功能

- **NLP日志解析**：使用BERT模型将非结构化日志转换为结构化数据
- **智能查询引擎**：通过自然语言输入直接查询日志，无需编写复杂SQL
- **异常检测**：结合监督学习和无监督学习，自动识别异常行为和攻击模式
- **自适应告警**：动态调整告警阈值，减少误报率
- **安全报告生成**：自动生成安全事件摘要和修复建议

## 技术架构

本项目由三个主要模块组成：

1. **前端**：负责Web UI界面和可视化展示
2. **后端**：提供API服务和数据处理功能
3. **AI服务**：负责NLP模型训练和推理

### AI模块技术栈

- **模型框架**：Hugging Face Transformers（BERT/GPT微调）
- **深度学习库**：PyTorch Lightning
- **模型部署**：ONNX Runtime
- **数据处理**：Pandas、PySpark
- **实验管理**：MLflow

## 项目结构
项目目录结构如下所示：

```
app/
  ├── __init__.py
  ├── app.py                 # 应用入口
  ├── models/                # 数据模型
  │   ├── __init__.py
  │   ├── db.py              # 数据库定义
  │   ├── log_model.py       # 日志模型
  │   └── anomaly_model.py   # 异常模型
  ├── routes/                # API路由
  │   ├── __init__.py
  │   ├── log_routes.py      # 日志API
  │   ├── nlp_routes.py      # NLP分析API
  │   └── anomaly_routes.py  # 异常检测API
  ├── services/              # 业务服务
  │   ├── __init__.py
  │   ├── log_parser.py      # 日志解析服务
  │   ├── nlp_processor.py   # NLP处理服务
  │   └── anomaly_detector.py# 异常检测服务
  ├── static/                # 静态资源
  │   ├── css/
  │   ├── js/
  │   └── images/
  └── templates/             # 前端模板
      ├── layout.html
      ├── index.html
      └── dashboard.html
```

本项目基于flask结构，在docker上集成环境，因此环境不需额外配置。运行run.py即可在本地5000端口上运行可视化界面。

## 开发指南 (AI模块)

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/yourusername/NLP-SecLogAI.git
cd NLP-SecLogAI/ai

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 模型训练

```bash
cd training
python train.py --config configs/bert_classifier.yaml
```

### 启动推理服务

```bash
cd inference
uvicorn app:app --reload --port 8001
```

## API接口

AI模块提供以下关键API接口：

1. **自然语言转SQL**
   ```
   POST /ai/nl2sql
   {"query": "最近24小时admin登录失败次数"}
   ```

2. **日志异常检测**
   ```
   POST /ai/detect_anomaly
   {"log_text": "User admin failed to login from 192.168.1.100"}
   ```

3. **安全报告生成**
   ```
   POST /ai/generate_report
   {"logs": [...]}
   ```

## 创新点

本项目相比传统SIEM系统具有以下创新点：

1. **基于NLP的智能日志解析**：不再依赖关键字匹配，利用BERT进行语义分析
2. **智能日志查询**：用户可用自然语言查询安全事件，降低使用门槛
3. **自适应AI告警优化**：系统会基于用户反馈自动调整告警阈值，降低误报
4. **GPT生成安全报告**：AI自动提炼日志内容，生成可读的安全分析报告

## 使用示例

### 自然语言查询

用户输入：
```
过去24小时内admin用户登录失败了多少次？
```

系统转换为SQL并执行：
```sql
SELECT COUNT(*) as failure_count 
FROM logs 
WHERE timestamp >= NOW() - INTERVAL '24 HOUR' 
AND status = 'failure' 
AND event_type = 'logon' 
AND user = 'admin'
```

### 异常检测

系统自动检测出可疑行为：
1. 短时间内多次登录失败
2. 敏感文件访问
3. 异常端口连接尝试

## 贡献指南

欢迎贡献代码或提出问题！请提交 Pull Request 或创建 Issue。

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

# TinyLogBERT 模型改进

本次更新对日志异常检测模型 TinyLogBERT 进行了多项增强，主要包括以下改进：

## 1. 动量对比学习（基于MoCo架构）

- **动量编码器（Momentum Encoder）**：
  - 添加了动量编码器作为"key encoder"，降低对比学习中的特征不一致性
  - 实现平滑更新策略，帮助模型更稳定地学习表示

- **特征队列（Queue）**：
  - 维护一个大小为4096的特征队列，提供更多的负样本
  - 采用先进先出策略动态更新队列，使模型能看到更多样本

## 2. 多视角正样本对生成

- **数据增强策略**：
  - 实现了三种增强策略：随机删除、随机替换和随机打乱
  - 对同一日志生成语义相似但表达不同的变体，作为正样本对

## 3. 可训练温度参数

- 将对比学习中的温度参数τ设计为可学习参数
- 引入对温度参数的约束，确保其在合理范围内（0.05至0.5）
- 实现自适应调整，使模型能够自动找到最佳的对比强度

## 4. 增强的异常评分机制

- **融合对比距离**：
  - 将样本与队列中样本的对比距离纳入异常评分
  - 使用加权融合策略，结合传统异常分数和对比距离
  - 显著提高了对异常样本的检测能力

## 5. 改进的训练流程

- **自定义数据集类**：实现支持对比学习的`ContrastiveLogDataset`
- **自定义数据校对器**：实现`ContrastiveMLMCollator`处理增强数据
- **自定义训练器**：实现`ContrastiveTrainer`处理MoCo风格训练逻辑

## 性能改进

- 更准确的异常检测结果，尤其对于复杂和边界情况
- 更稳定的训练过程，减少了表示学习中的波动
- 增强了模型对未见日志模式的泛化能力

## 使用示例

```python
# 创建并训练改进后的模型
detector = AnomalyDetector()
detector.train(
    train_file="logs/train.log",
    eval_file="logs/eval.log",
    enable_contrastive=True,  # 启用对比学习
    enable_augmentation=True  # 启用数据增强
)

# 评估模型性能
results = detector.evaluate("logs/test.jsonl")
print(f"AUC: {results['auc']}")
print(f"对比学习AUC: {results.get('contrastive_auc', 'N/A')}")
```

