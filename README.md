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

```
project-root/
├── frontend/               # 前端工程 (React + Next.js)
├── backend/                # 后端工程 (FastAPI)
├── ai/                     # AI工程 (负责人: C)
│   ├── training/           # 模型训练代码
│   │   ├── data/           # 训练数据和预处理
│   │   ├── models/         # 模型定义
│   │   ├── utils/          # 工具函数
│   │   └── train.py        # 训练入口
│   ├── inference/          # 模型推理服务
│   │   ├── api/            # FastAPI接口定义
│   │   ├── models/         # 模型加载和推理
│   │   └── app.py          # API服务入口
│   └── notebooks/          # Jupyter笔记本（实验用）
│       ├── log_analysis.ipynb
│       └── model_eval.ipynb
├── docker/                 # Docker部署配置
│   ├── frontend.Dockerfile
│   ├── backend.Dockerfile
│   └── ai.Dockerfile
├── docs/                   # 文档
└── README.md
```

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
