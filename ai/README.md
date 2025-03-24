# NLP-SecLogAI AI模块

这是NLP-SecLogAI项目的AI模块，负责安全日志的智能分析、异常检测和报告生成。

## 目录结构

```
ai/
├── training/           # 模型训练代码
│   ├── data/           # 训练数据和预处理
│   ├── models/         # 模型定义
│   ├── utils/          # 工具函数
│   ├── configs/        # 配置文件
│   ├── checkpoints/    # 模型检查点
│   ├── logs/           # 训练日志
│   └── train.py        # 训练入口
├── inference/          # 模型推理服务
│   ├── api/            # FastAPI接口定义
│   ├── models/         # 模型加载和推理
│   └── app.py          # API服务入口
├── notebooks/          # Jupyter笔记本（实验用）
│   ├── log_analysis.ipynb
│   └── model_eval.ipynb
└── requirements.txt    # Python依赖
```

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

1. **训练模型**:
```bash
cd training
python train.py --config configs/bert_classifier.yaml
```

2. **启动推理服务**:
```bash
cd inference
uvicorn app:app --reload --port 8001
```

## 主要功能

- 日志分类：识别不同类型的安全事件
- 异常检测：发现异常行为和攻击模式
- 自然语言查询：将自然语言转换为SQL查询
- 安全报告生成：自动生成安全事件分析报告

## 数据

- `data/sample_logs.txt`: 示例日志文件，用于开发和测试
- `data/logs_train.csv`: 训练数据集
- `data/logs_test.csv`: 测试数据集

## API文档

启动推理服务后，访问 `http://localhost:8001/docs` 查看API文档。

主要端点：
- `/ai/nl2sql`: 自然语言转SQL查询
- `/ai/detect_anomaly`: 日志异常检测
- `/ai/generate_report`: 安全报告生成

## 开发指南

详细的开发文档请参阅 `docs/ai_module.md`。 