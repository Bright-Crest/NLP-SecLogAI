# AI模块详细说明

## 架构概述

AI模块是NLP-SecLogAI系统的核心组件，负责对安全日志进行智能分析、异常检测和报告生成。整体架构基于现代深度学习技术构建，特别是利用BERT等预训练语言模型对日志文本进行语义理解。

## 子模块说明

### 1. 训练模块 (training/)

训练模块负责处理数据和训练机器学习模型，包括：

#### 数据处理 (data/)
- `log_dataset.py`: 定义用于训练的数据集类
- `sample_logs.txt`: 示例日志数据
- `logs_train.csv`: 训练集
- `logs_test.csv`: 测试集

#### 模型定义 (models/)
- `log_classifier.py`: 基于BERT的日志分类器，识别不同类型的日志事件
- 支持的日志类型包括：登录事件、注销事件、网络连接拦截、文件访问、系统错误等

#### 工具函数 (utils/)
- `preprocessing.py`: 日志清洗和预处理功能

#### 配置文件 (configs/)
- `bert_classifier.yaml`: 模型训练配置

#### 入口文件
- `train.py`: 训练入口，支持命令行参数配置

### 2. 推理模块 (inference/)

推理模块负责加载训练好的模型并提供API服务，包括：

#### API接口 (api/)
- `nl2sql.py`: 自然语言转SQL查询功能
- `report_generator.py`: 安全报告生成器

#### 模型加载 (models/)
- `model_loader.py`: 加载和初始化训练好的模型

#### 入口文件
- `app.py`: FastAPI服务入口

### 3. 实验笔记本 (notebooks/)

包含用于数据探索和模型评估的Jupyter笔记本：

- `log_analysis.ipynb`: 日志分析与可视化
- `model_eval.ipynb`: 模型性能评估

## 功能特点

### 1. 日志分类

- 基于BERT的多分类模型，将日志分为5类：
  - 登录事件 (logon)
  - 注销事件 (logoff)
  - 网络连接 (connection_blocked)
  - 文件访问 (file_access)
  - 系统错误 (system_error)

### 2. 异常检测

- 结合监督学习和无监督学习方法
- 可以检测以下类型的异常：
  - 频繁登录失败
  - 敏感文件访问
  - 异常网络连接
  - 异常系统行为

### 3. 自然语言查询

- 支持用自然语言直接查询日志
- 自动将自然语言转换为SQL查询
- 示例：
  - "过去24小时admin登录失败次数"
  - "最近一周来自外部IP的连接拦截"

### 4. 安全报告生成

- 自动生成安全事件摘要
- 提供威胁等级评估
- 生成安全建议和修复方案

## 使用指南

### 训练新模型

```bash
cd ai/training
python train.py --config configs/bert_classifier.yaml
```

### 评估模型性能

```bash
cd ai/notebooks
jupyter notebook model_eval.ipynb
```

### 启动推理服务

```bash
cd ai/inference
uvicorn app:app --reload --port 8001
```

### API调用示例

1. 自然语言转SQL：
```
curl -X POST "http://localhost:8001/ai/nl2sql" \
     -H "Content-Type: application/json" \
     -d '{"query": "最近24小时admin登录失败次数"}'
```

2. 异常检测：
```
curl -X POST "http://localhost:8001/ai/detect_anomaly" \
     -H "Content-Type: application/json" \
     -d '{"log_text": "User admin failed to login from 192.168.1.100"}'
```

## 性能指标

- 日志分类准确率：~92%
- 异常检测F1分数：~0.85
- 自然语言查询准确率：~90%
- 推理延迟：<100ms

## 扩展开发

1. 添加新的日志类型：
   - 在`training/data`添加新类型的样本
   - 修改`log_classifier.py`中的分类数量
   - 重新训练模型

2. 改进异常检测：
   - 添加新的异常检测规则
   - 考虑集成更复杂的时序异常检测算法

3. 增强报告生成：
   - 针对不同类型的攻击生成专门的修复建议
   - 添加漏洞参考信息

## 局限性

- 依赖高质量的训练数据
- 对新型、未见过的攻击模式识别能力有限
- 日志格式变化可能影响解析准确性

## 未来计划

- 集成更多类型的日志源（如云服务日志、IoT设备日志）
- 添加更多语言支持
- 实现主动学习机制，通过用户反馈持续优化模型
- 增加攻击溯源和攻击图构建功能 