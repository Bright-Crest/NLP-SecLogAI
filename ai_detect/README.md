# 🚀 TinyLogBERT - 无盒标日志异常检测系统

基于BERT-mini的轻量级模型，用于无监督日志异常检测，无需手动标记的简单高效系统。

## 📋 功能特点

- **轻量级模型**：基于BERT-mini，适合在普通硬件上运行
- **无盒标检测**：无需大量人工标记数据
- **易于扩展**：模块化设计，可轻松替换或升级各组件
- **REST API**：提供Flask API接口，方便集成
- **CLI工具**：命令行工具，方便本地批处理
- **效果量化**：提供ROC-AUC、PR曲线等评估指标
- **可视化支持**：支持分数分布、t-SNE可视化等
- **模型对比**：支持TinyLogBERT与LogBERT-Distil的性能对比
- **日志自动转换**：支持将普通日志文件自动转换为训练所需的JSON格式

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

## 📝 模型微调

### TinyLogBERT模型训练

```bash
python train.py --data-file ./data/hdfs_logs.jsonl --model-type tinylogbert --batch-size 32 --epochs 5
```

### LogBERT-Distil模型训练

```bash
python train.py --data-file ./data/hdfs_logs.jsonl --model-type logbert_distil --batch-size 32 --epochs 5
```

### 使用普通日志文件训练

系统支持直接使用普通日志文件进行训练，会自动转换为所需的JSON格式：

```bash
python train.py --data-file ./data/raw_logs.log --anomaly-ratio 0.05 --model-type tinylogbert
```

参数说明：
- `--data-file`: 训练数据文件路径（支持普通日志文件或JSON格式）
- `--model-type`: 模型类型 (tinylogbert 或 logbert_distil)
- `--sample-limit`: 限制训练样本数量
- `--batch-size`: 批大小
- `--epochs`: 训练轮数
- `--learning-rate`: 学习率
- `--checkpoint-dir`: 模型保存目录
- `--anomaly-ratio`: 普通日志文件转换时的异常比例（默认0.05）

## 🔄 日志格式转换工具

系统提供单独的日志转换工具，可以将普通日志文件转换为训练所需的JSON格式：

```bash
python log_converter.py --input ./data/raw_logs.log
```

该工具会自动将日志文件转换为JSON格式并保存，并智能检测是否已存在处理过的文件，避免重复处理。

### 分析日志文件

在转换前可以先分析日志文件，了解其基本特征：

```bash
python log_converter.py --input ./data/raw_logs.log --analyze
```

### 转换参数说明

- `--input`, `-i`: 输入日志文件路径
- `--output`, `-o`: 输出JSON文件路径（默认为输入文件名加_jsonl.json）
- `--anomaly-ratio`, `-r`: 模拟的异常比例（默认0.05）
- `--force`, `-f`: 强制转换，即使输出文件已存在
- `--analyze`, `-a`: 仅分析日志，不进行转换
- `--seed`, `-s`: 随机种子，确保结果可复现

## 🔍 异常检测 API

启动API服务：

```bash
python run.py
```

API端点：
- `POST /api/anomaly/detect` - 检测单条日志是否异常
- `POST /api/anomaly/batch_detect` - 批量检测多条日志
- `POST /api/anomaly/tokenize` - 将日志文本转换为token列表

### 使用示例

检测单条日志：

```bash
curl -X POST http://localhost:5000/api/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{"log_text": "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"}'
```

## 🖥️ 命令行工具

检测单条日志：

```bash
python ai_detect_cli.py detect "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"
```

批量检测日志文件：

```bash
python ai_detect_cli.py batch logs.txt --output results.json
```

### 日志格式转换

直接将普通日志文件转换为JSON格式：

```bash
python ai_detect_cli.py convert logs.txt --anomaly-ratio 0.05
```

### 命令行工具参数说明

**detect** - 检测单条日志：
- `log` - 日志文本
- `-t`, `--threshold` - 异常检测阈值(0-1)

**batch** - 批量检测日志文件：
- `file` - 日志文件路径
- `-t`, `--threshold` - 异常检测阈值(0-1)
- `-o`, `--output` - 输出结果文件路径
- `--json-format` - 指定输入文件为JSON格式

**convert** - 转换日志文件：
- `file` - 日志文件路径
- `-o`, `--output` - 输出JSON文件路径
- `-r`, `--anomaly-ratio` - 模拟的异常比例(0-1)
- `-f`, `--force` - 强制转换

**tokenize** - 将日志转换为token列表：
- `log` - 日志文本

查看帮助：

```bash
python ai_detect_cli.py --help
```

## 📊 效果评估

评估模型效果：

```python
from app.services.evaluate import evaluate_scores, visualize_results

# 评估
scores = [0.2, 0.8, 0.3, 0.9, 0.1]  # 模型预测分数
labels = [0, 1, 0, 1, 0]  # 真实标签（0:正常，1:异常）
auc = evaluate_scores(scores, labels)

# 可视化
visualize_results(scores, labels)
```

## 📈 模型对比

比较TinyLogBERT和LogBERT-Distil模型性能：

```bash
python model_comparison_example.py --test-data ./data/test_logs.jsonl \
  --tiny-model-path ./checkpoint/tiny_model.pt \
  --distil-model-path ./checkpoint/distil_model.pt
```

比较结果包括：
- ROC曲线对比
- 精确率-召回率曲线对比
- 推理时间对比
- 嵌入向量可视化对比

## 🚀 完整工作流程示例

系统提供了完整的工作流程示例脚本，从原始日志到模型训练、评估、对比的全流程：

```bash
python workflow_example.py --raw-logs ./data/raw_logs.log --epochs 3 --batch-size 32
```

参数说明：
- `--raw-logs`: 原始日志文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批大小
- `--sample-limit`: 样本数量限制
- `--output-dir`: 输出目录

这个脚本会执行以下步骤：
1. 分析日志文件
2. 转换日志文件为JSON格式
3. 划分训练集和测试集
4. 训练TinyLogBERT模型
5. 训练LogBERT-Distil模型
6. 比较两个模型的性能
7. 使用CLI工具进行异常检测

结果将保存在指定的输出目录中，包括：
- 转换后的日志文件
- 训练集和测试集
- 模型检查点
- 性能比较结果
- 异常检测结果

## 🔧 项目结构

```
TinyLogBERT/
│── app/                           # 核心应用目录
│   ├── models/                     # 模型定义
│   │   ├── tinylogbert_model.py     # TinyLogBERT模型
│   │   ├── logbert_distil_model.py  # LogBERT-Distil模型
│   ├── services/                    # 业务逻辑
│   │   ├── log_tokenizer.py          # 日志转token分词器
│   │   ├── anomaly_score.py          # 异常分数计算
│   │   ├── evaluate.py               # 效果评估模块
│   │   ├── model_comparison.py       # 模型对比服务
│   ├── routes/                      # API路由
│       ├── anomaly_routes.py         # 异常检测API
│── ai_detect_cli.py                 # 命令行工具
│── run.py                          # API服务启动脚本
│── train.py                        # 模型训练脚本
│── log_converter.py                # 日志格式转换工具
│── model_comparison_example.py     # 模型对比示例脚本
│── workflow_example.py             # 完整工作流程示例脚本
│── config.py                       # 配置文件
│── requirements.txt                # 依赖列表
```

## 🚀 未来计划

- [ ] 支持更多日志格式
- [ ] 添加自动阈值优化
- [ ] 集成FastAPI提高性能
- [ ] 支持分布式部署
- [ ] 添加简单Web界面

## 📄 许可证

MIT 