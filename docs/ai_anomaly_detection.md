# NLP-SecLogAI: 基于轻量级BERT的日志异常检测系统

## 目录

- [NLP-SecLogAI: 基于轻量级BERT的日志异常检测系统](#nlp-seclogai-基于轻量级bert的日志异常检测系统)
  - [目录](#目录)
  - [系统概述](#系统概述)
    - [总体架构设计](#总体架构设计)
      - [架构图](#架构图)
      - [架构说明](#架构说明)
    - [系统边界与模块划分](#系统边界与模块划分)
      - [核心组件](#核心组件)
      - [系统边界](#系统边界)
    - [关键技术选型与理由](#关键技术选型与理由)
    - [系统运行环境要求](#系统运行环境要求)
      - [硬件要求](#硬件要求)
      - [软件要求](#软件要求)
      - [中间件要求](#中间件要求)
  - [功能设计](#功能设计)
    - [模块功能描述](#模块功能描述)
      - [1. 数据处理模块](#1-数据处理模块)
      - [2. AI模型模块](#2-ai模型模块)
      - [3. 服务接口模块](#3-服务接口模块)
    - [流程图/状态图/时序图](#流程图状态图时序图)
      - [1. 训练流程图](#1-训练流程图)
      - [2. 检测流程图](#2-检测流程图)
      - [3. 系统时序图](#3-系统时序图)
      - [4. 模型训练状态图](#4-模型训练状态图)
    - [接口定义](#接口定义)
      - [1. 模块间接口](#1-模块间接口)
      - [2. 对外接口](#2-对外接口)
        - [REST API接口](#rest-api接口)
        - [命令行接口](#命令行接口)
  - [详细设计](#详细设计)
    - [数据库设计](#数据库设计)
      - [文件存储结构](#文件存储结构)
      - [异常记录存储](#异常记录存储)
    - [类图/对象模型](#类图对象模型)
      - [主要类职责](#主要类职责)
    - [算法逻辑说明](#算法逻辑说明)
      - [1. 无监督异常检测算法](#1-无监督异常检测算法)
        - [KNN检测器](#knn检测器)
        - [聚类检测器](#聚类检测器)
        - [重构损失检测器](#重构损失检测器)
      - [2. 动态融合算法](#2-动态融合算法)
      - [3. MLM与对比学习结合的训练算法](#3-mlm与对比学习结合的训练算法)
      - [4. KNN增强的实时检测算法](#4-knn增强的实时检测算法)
    - [核心代码结构与关键实现说明](#核心代码结构与关键实现说明)
      - [1. 项目结构](#1-项目结构)
      - [2. TinyLogBERT模型实现](#2-tinylogbert模型实现)
      - [3. 异常检测服务实现](#3-异常检测服务实现)
      - [4. API接口实现](#4-api接口实现)
  - [接口文档](#接口文档)
    - [接口路径](#接口路径)
    - [请求方法](#请求方法)
      - [1. `/ai/score_log` 接口](#1-aiscore_log-接口)
      - [2. `/ai/score_log_sequence` 接口](#2-aiscore_log_sequence-接口)
      - [3. `/ai/detect` 接口](#3-aidetect-接口)
      - [4. `/ai/model/status` 接口](#4-aimodelstatus-接口)
      - [5. `/ai/model/threshold` 接口](#5-aimodelthreshold-接口)
      - [6. `/ai/knn/build` 接口](#6-aiknnbuild-接口)
      - [7. `/ai/knn/status` 接口](#7-aiknnstatus-接口)
      - [8. `/ai/ui` 接口](#8-aiui-接口)
    - [请求参数](#请求参数)
      - [1. `/ai/score_log` 接口](#1-aiscore_log-接口-1)
      - [2. `/ai/score_log_sequence` 接口](#2-aiscore_log_sequence-接口-1)
      - [3. `/ai/detect` 接口](#3-aidetect-接口-1)
      - [4. `/ai/model/status` 接口](#4-aimodelstatus-接口-1)
      - [5. `/ai/model/threshold` 接口](#5-aimodelthreshold-接口-1)
      - [6. `/ai/knn/build` 接口](#6-aiknnbuild-接口-1)
      - [7. `/ai/knn/status` 接口](#7-aiknnstatus-接口-1)
    - [返回值结构](#返回值结构)
      - [1. `/ai/score_log` 接口](#1-aiscore_log-接口-2)
      - [2. `/ai/score_log_sequence` 接口](#2-aiscore_log_sequence-接口-2)
      - [3. `/ai/detect` 接口](#3-aidetect-接口-2)
      - [4. `/ai/model/status` 接口](#4-aimodelstatus-接口-2)
      - [5. `/ai/model/threshold` 接口](#5-aimodelthreshold-接口-2)
      - [6. `/ai/knn/build` 接口](#6-aiknnbuild-接口-2)
      - [7. `/ai/knn/status` 接口](#7-aiknnstatus-接口-2)
    - [状态码说明](#状态码说明)
    - [调用示例](#调用示例)
      - [Python示例](#python示例)
      - [cURL示例](#curl示例)
      - [JavaScript示例](#javascript示例)

## 系统概述

### 总体架构设计

#### 架构图

```
+---------------------+     +---------------------+     +----------------------+
|                     |     |                     |     |                      |
|    数据处理层       |---->|    AI模型层         |---->|    服务接口层        |
|                     |     |                     |     |                      |
+---------------------+     +---------------------+     +----------------------+
         |                           |                           |
         v                           v                           v
+---------------------+     +---------------------+     +----------------------+
| • 日志预处理        |     | • TinyLogBERT模型   |     | • RESTful API        |
| • 日志解析          |     | • 无监督学习模块    |     | • 命令行接口         |
| • 窗口处理          |     | • 异常评分融合      |     | • Web界面            |
| • 特征提取          |     | • 特征记忆库        |     | • 可视化组件         |
+---------------------+     +---------------------+     +----------------------+
```

#### 架构说明

NLP-SecLogAI系统采用分层架构设计，由三个主要层次组成：

1. **数据处理层**：负责日志数据的预处理、解析和转换
   - 支持多种日志格式和编码
   - 实现固定窗口和滑动窗口处理
   - 提供实时数据流处理和批处理功能

2. **AI模型层**：核心智能分析引擎
   - 基于轻量级BERT (TinyLogBERT) 的自监督学习模型
   - 多种无监督异常检测算法集成
   - 动态融合的异常评分系统
   - 特征记忆库和KNN增强机制

3. **服务接口层**：对外提供服务的接口
   - RESTful API接口
   - 命令行工具
   - Web可视化界面
   - 实时监控与报警功能

这种分层架构确保了系统的灵活性和可扩展性，各层之间通过明确的接口进行交互，支持组件的独立开发和替换。

### 系统边界与模块划分

#### 核心组件

1. **日志处理模块**
   - `LogWindow`: 处理日志窗口分段
   - `LogTokenizer`: 日志标记化与预处理

2. **AI模型模块**
   - `TinyLogBERT`: 轻量级BERT模型核心
   - `AnomalyDetector`: 异常检测引擎
   - `DynamicLossWeighting`: 动态损失权重调整

3. **检测方法模块**
   - `UnsupervisedAnomalyDetector`: 无监督异常检测器
   - `AnomalyScoreEnsemble`: 异常评分融合系统
   - `ReconstructionHead`: 重构损失评估模块

4. **服务模块**
   - `AnomalyScoreService`: 异常评分服务
   - `API Routes`: RESTful接口实现
   - `CLI Tool`: 命令行接口

#### 系统边界

- **输入边界**：日志文本（单行或序列）
- **输出边界**：异常评分、异常窗口位置、可视化结果
- **外部依赖**：
  - PyTorch框架（模型训练与推理）
  - Flask（API服务）
  - Transformers库（预训练模型基础）

### 关键技术选型与理由

1. **BERT-mini作为基础模型**
   - **理由**：相比完整BERT模型(110M参数)，BERT-mini仅有11M参数，大幅减少资源占用，同时保留了足够的语义理解能力，适合日志场景的轻量级部署
   - **优势**：推理速度提升10倍以上，内存占用减少90%，适合边缘设备部署

2. **无监督学习方法**
   - **理由**：安全日志中异常样本稀少且多样，有标注数据难以获取，无监督学习可以仅基于正常数据进行训练
   - **创新**：结合MLM自监督学习和对比学习，从多个角度学习正常日志的表示

3. **多窗口策略**
   - **理由**：日志异常通常跨越多行且有上下文依赖，窗口处理能捕捉到行为序列模式
   - **优势**：同时支持固定窗口（效率高）和滑动窗口（精度高）两种处理方式

4. **异常检测方法融合**
   - **理由**：单一检测方法往往偏向特定类型的异常，多方法融合可提高检测覆盖面
   - **创新点**：引入动态权重策略，根据样本特征自适应调整各检测器权重

5. **KNN增强机制**
   - **理由**：为模型引入记忆能力，使其能记住"正常"的行为模式
   - **优势**：提高检测精度，减少误报，并能从运行时数据中持续学习

6. **PyTorch框架**
   - **理由**：提供灵活的深度学习模型定义和优化功能
   - **优势**：广泛的社区支持，良好的GPU加速，动态计算图支持

### 系统运行环境要求

#### 硬件要求

- **最低配置**：
  - CPU: 双核处理器，主频2.0GHz以上
  - 内存: 4GB RAM
  - 磁盘空间: 1GB可用空间

- **推荐配置**：
  - CPU: 四核处理器，主频3.0GHz以上
  - GPU: NVIDIA GPU，支持CUDA
  - 内存: 4GB RAM以上
  - 磁盘空间: 10GB可用空间（含日志存储）

#### 软件要求

- **操作系统**：
  - Linux(Ubuntu 18.04+, CentOS 7+)
  - Windows 10/11
  - macOS 10.15+

- **运行时环境**：
  - Python 3.8+
  - PyTorch 1.10+
  - CUDA 11.0+（GPU加速，可选）

- **依赖库**：
  - transformers 4.15+
  - flask 2.0+
  - numpy 1.20+
  - scikit-learn 1.0+
  - pandas 1.3+
  - matplotlib 3.4+

#### 中间件要求

- Web服务器部署时推荐：
  - Nginx 1.18+
  - uWSGI 2.0+（Python应用服务器）

## 功能设计

### 模块功能描述

#### 1. 数据处理模块

数据处理模块主要负责日志数据的预处理、转换和窗口化，为AI模型提供结构化输入。

| 子模块 | 功能描述 |
|-------|---------|
| **LogWindow** | 实现日志窗口分段策略，支持固定窗口和滑动窗口两种模式；负责将连续日志切分为有意义的行为序列 |
| **LogTokenizer** | 日志标记化处理，将原始日志文本转换为模型可处理的token序列；支持特殊字符处理和标准化 |
| **数据增强** | 通过同义词替换、词序调整等技术生成相似但有变化的日志样本，增强模型鲁棒性 |
| **编码处理** | 支持多种编码格式的日志文件读取，自动检测和转换，确保数据完整性 |

#### 2. AI模型模块

AI模型模块是系统的核心，基于轻量级BERT架构实现日志语义理解和异常检测。

| 子模块 | 功能描述 |
|-------|---------|
| **TinyLogBERT** | 轻量级BERT模型实现，包含4层Transformer编码器，专为日志文本优化；同时提供MLM和对比学习训练目标 |
| **DynamicLossWeighting** | 动态损失权重调整机制，支持基于不确定性和梯度范数的两种权重策略，自动平衡多任务学习目标 |
| **AnomalyScoreEnsemble** | 异常评分融合系统，集成多种异常检测方法的评分结果，通过动态权重提高检测准确率 |
| **UnsupervisedAnomalyDetector** | 无监督异常检测组件，实现KNN、聚类、局部离群因子等多种无监督异常检测算法 |
| **MomentumEncoder** | 动量编码器实现，基于MoCo设计，维护特征一致性并减少模型更新震荡 |
| **ReconstructionHead** | 自编码重构头，用于重构输入序列，通过重构损失评估样本异常程度 |

#### 3. 服务接口模块

服务接口模块将AI能力包装为易用的API和工具，支持与其他系统集成。

| 子模块 | 功能描述 |
|-------|---------|
| **AnomalyScoreService** | 异常评分服务，提供统一的异常检测接口，支持单条日志和批量日志评分 |
| **RESTful API** | HTTP接口实现，提供标准化的异常检测服务，支持JSON格式交互 |
| **CLI工具** | 命令行接口工具，支持文件处理、批量检测和结果导出，适合脚本集成 |
| **Web界面** | 基于Bootstrap的Web可视化界面，提供直观的交互和结果展示 |
| **模型管理** | 模型加载、保存和版本管理功能，支持多模型切换和热更新 |

### 流程图/状态图/时序图

#### 1. 训练流程图

```
+---------------+     +----------------+     +-------------------+
| 加载正常日志  |---->| 创建日志窗口   |---->| 构建训练数据集    |
+---------------+     +----------------+     +-------------------+
                                                      |
                                                      v
+------------------+     +------------------+     +-------------------+
| 保存最终模型     |<----| 验证模型性能     |<----| MLM+对比学习训练  |
+------------------+     +------------------+     +-------------------+
        |
        v
+------------------+     +------------------+     +-------------------+
| 特征记忆库构建   |---->| 无监督检测器训练 |---->| 评分融合器优化    |
+------------------+     +------------------+     +-------------------+
```

#### 2. 检测流程图

```
+---------------+     +----------------+     +-------------------+
| 接收日志输入  |---->| 日志预处理     |---->| 创建窗口表示      |
+---------------+     +----------------+     +-------------------+
                                                      |
                                                      v
+------------------+     +------------------+     +-------------------+
| 返回检测结果     |<----| 异常判定与分析   |<----| 多方法异常评分    |
+------------------+     +------------------+     +-------------------+
```

#### 3. 系统时序图

```
+------------+     +------------+     +------------+     +------------+
|   客户端   |     |  API服务   |     | 模型服务   |     | 异常检测   |
+------------+     +------------+     +------------+     +------------+
      |                  |                  |                  |
      | 发送日志         |                  |                  |
      |----------------->|                  |                  |
      |                  | 请求处理         |                  |
      |                  |----------------->|                  |
      |                  |                  | 加载模型         |
      |                  |                  |----------------->|
      |                  |                  |                  |
      |                  |                  | 执行评分         |
      |                  |                  |<-----------------|
      |                  | 返回评分结果     |                  |
      |                  |<-----------------|                  |
      | 返回检测结果     |                  |                  |
      |<-----------------|                  |                  |
      |                  |                  |                  |
```

#### 4. 模型训练状态图

```
+---------------+     +----------------+     +-------------------+
|    初始化     |---->| 数据预处理     |---->| MLM预训练阶段     |
+---------------+     +----------------+     +-------------------+
      ^                                              |
      |                                              v
+------------------+     +------------------+     +-------------------+
| 失败/重试        |<----| 无监督检测器训练 |<----| 对比学习阶段      |
+------------------+     +------------------+     +-------------------+
                                 |
                                 v
                         +-------------------+
                         | 模型导出与保存    |
                         +-------------------+
                                 |
                                 v
                         +-------------------+
                         |   完成/就绪       |
                         +-------------------+
```

### 接口定义

#### 1. 模块间接口

| 接口名称 | 功能描述 | 输入参数 | 输出结果 |
|---------|---------|---------|---------|
| `LogWindow.create_fixed_windows` | 创建固定大小的日志窗口 | 日志列表, 窗口大小 | 窗口表示, 窗口文本 |
| `LogWindow.create_sliding_windows` | 创建滑动窗口的日志窗口 | 日志列表, 步长 | 窗口表示列表 |
| `AnomalyDetector.detect` | 检测单条日志是否异常 | 日志文本, 阈值 | 异常分数, 异常标志 |
| `AnomalyDetector.detect_sequence` | 检测日志序列中的异常 | 日志列表, 窗口类型, 步长 | 窗口分数列表, 异常窗口 |
| `AnomalyScoreService.score_single_log` | 评估单条日志的异常分数 | 日志文本, KNN开关 | 异常分数 |
| `AnomalyScoreService.score_log_sequence` | 评估日志序列的异常分数 | 日志列表, 窗口类型, 步长 | 分数列表, 平均分数, 最大分数 |
| `TinyLogBERT.forward` | 模型前向传播 | input_ids, attention_mask | anomaly_score, cls_embedding, 其他输出 |

#### 2. 对外接口

系统通过REST API和命令行工具提供对外服务接口：

##### REST API接口

| 接口路径 | 方法 | 功能描述 |
|---------|------|---------|
| `/ai/score_log` | POST | 对单条日志进行异常评分 |
| `/ai/score_log_sequence` | POST | 对日志序列进行异常评分 |
| `/ai/detect` | POST | 使用异常检测器检测日志 |
| `/ai/model/status` | GET | 获取模型状态 |
| `/ai/model/threshold` | POST | 设置异常阈值 |
| `/ai/knn/build` | POST | 构建KNN嵌入库 |
| `/ai/knn/status` | GET/POST | 获取或设置KNN增强状态 |
| `/ai/ui` | GET | AI异常检测Web界面 |

##### 命令行接口

| 命令选项 | 功能描述 |
|---------|---------|
| `--logs` | 指定要检测的日志文本 |
| `--log-file` | 指定要检测的日志文件 |
| `--model-path` | 指定模型路径 |
| `--window-type` | 指定窗口类型(fixed/sliding) |
| `--threshold` | 指定异常判定阈值 |
| `--output` | 指定输出文件路径 |
| `--use-knn` | 是否使用KNN增强 |
| `--visualization` | 是否生成可视化结果 |

## 详细设计

### 数据库设计

本系统主要关注实时的日志异常检测，不依赖关系型数据库存储核心数据。但为了支持历史异常记录查询、模型管理和配置持久化，系统设计了以下简单的数据结构：

#### 文件存储结构

系统使用文件系统保存模型、配置和临时数据：

| 文件/目录 | 用途 | 存储内容 |
|---------|------|---------|
| `/ai_detect/checkpoint/` | 模型保存目录 | 预训练的TinyLogBERT模型文件、配置文件 |
| `/ai_detect/output/` | 输出结果目录 | 评估结果、可视化图表、性能报告 |
| `/app/config/` | 配置目录 | 系统配置文件、阈值设置、运行参数 |
| `/app/data/` | 数据存储 | 特征记忆库、KNN模型、异常记录 |

#### 异常记录存储

系统提供可选的异常记录存储功能，支持以下两种方式：

1. **JSON文件存储**：将检测到的异常以JSON格式保存在文件中
   ```json
   {
     "timestamp": "2023-04-05T12:34:56",
     "log": "failed login attempt from ip 192.168.1.10",
     "score": 0.92,
     "is_anomaly": true,
     "detection_method": "ensemble",
     "window_id": 12
   }
   ```

2. **外部数据库集成**：提供与外部数据库系统集成的连接器接口
   - 支持MongoDB用于半结构化日志存储
   - 支持PostgreSQL/MySQL用于关系型数据存储
   - 支持Elasticsearch用于大规模日志检索

### 类图/对象模型

系统核心模块的类图和对象关系如下：

```
+------------------+       +-------------------+        +-------------------+
| LogWindow        |<----->| AnomalyDetector   |<------>| TinyLogBERT       |
+------------------+       +-------------------+        +-------------------+
| - window_size    |       | - model           |        | - bert            |
| - tokenizer      |       | - tokenizer       |        | - anomaly_methods |
| - max_length     |       | - window          |        | - queue           |
+------------------+       | - detection_method|        | - anomaly_head    |
| + create_fixed   |       +-------------------+        +-------------------+
| + create_sliding |       | + detect          |        | + forward         |
| + batch_windows  |       | + detect_sequence |        | + get_cls_emb     |
+------------------+       | + train           |        | + _dequeue_enqueue|
                           | + evaluate        |        +-------------------+
                           +-------------------+                 ^
                                    ^                           |
                                    |                           |
+-------------------+      +-------------------+       +-------------------+
| AnomalyScoreService|<--->| LogDataset        |       | UnsupervisedAD    |
+-------------------+      +-------------------+       +-------------------+
| - detector        |      | - tokenizer       |       | - method          |
| - threshold       |      | - logs            |       | - n_neighbors     |
| - use_knn         |      | - max_length      |       | - reference_emb   |
+-------------------+      +-------------------+       | - model           |
| + score_single_log|      | + __getitem__     |       +-------------------+
| + score_sequence  |      | + __len__         |       | + fit             |
| + build_emb_bank  |      +-------------------+       | + get_anomaly_score|
+-------------------+                                  +-------------------+
        ^                                                      ^
        |                                                      |
        v                                                      v
+-------------------+                                 +-------------------+
| API Routes        |                                 | AnomalyScoreEnsemble|
+-------------------+                                 +-------------------+
| + score_log       |                                 | - detectors       |
| + score_sequence  |                                 | - weights         |
| + detect          |                                 | - fusion_method   |
| + model_status    |                                 +-------------------+
| + set_threshold   |                                 | + forward         |
| + build_knn       |                                 | + set_weights     |
+-------------------+                                 +-------------------+
```

#### 主要类职责

1. **LogWindow**：日志窗口处理类
   - 负责将原始日志序列处理为模型可用的窗口表示
   - 支持固定窗口和滑动窗口两种策略
   - 提供批处理功能，优化计算效率

2. **AnomalyDetector**：异常检测器类
   - 系统的主控制器，协调各组件工作
   - 提供模型训练、评估和检测功能
   - 管理检测方法和阈值设置

3. **TinyLogBERT**：轻量级BERT模型类
   - 继承自BertForMaskedLM，专为日志检测优化
   - 集成多种异常检测方法
   - 提供队列机制进行特征管理

4. **UnsupervisedAnomalyDetector**：无监督异常检测类
   - 实现多种无监督异常检测算法
   - 管理参考特征库
   - 提供异常分数计算接口

5. **AnomalyScoreService**：异常评分服务类
   - 对外提供统一的评分接口
   - 管理检测阈值和KNN功能状态
   - 支持单条和批量日志处理

6. **AnomalyScoreEnsemble**：评分融合类
   - 组合多种检测方法的结果
   - 实现动态权重融合策略
   - 优化整体检测性能

### 算法逻辑说明

#### 1. 无监督异常检测算法

系统实现了多种无监督异常检测算法，适应不同类型的异常：

##### KNN检测器

基于K近邻的异常检测，通过计算样本到其K个最近邻的平均距离作为异常分数：

```python
def knn_detector(features, reference_embeddings, n_neighbors=5):
    # 创建KNN模型
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(reference_embeddings)
    
    # 计算距离
    distances, _ = knn.kneighbors(features)
    
    # 计算异常分数
    mean_distances = np.mean(distances, axis=1)
    
    # 归一化分数
    max_dist = np.max(mean_distances) if len(mean_distances) > 0 else 1.0
    scores = mean_distances / (max_dist + 1e-10)
    
    return scores
```

##### 聚类检测器

基于K-means聚类的异常检测，通过样本到最近聚类中心的距离作为异常分数：

```python
def cluster_detector(features, reference_embeddings, n_clusters=10):
    # 创建聚类模型
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(reference_embeddings)
    
    # 计算样本到最近聚类中心的距离
    distances = np.min(
        [np.linalg.norm(features - center, axis=1) for center in kmeans.cluster_centers_], 
        axis=0
    )
    
    # 归一化分数
    max_dist = np.max(distances) if len(distances) > 0 else 1.0
    scores = distances / (max_dist + 1e-10)
    
    return scores
```

##### 重构损失检测器

基于自编码器的异常检测，通过输入重构损失作为异常分数：

```python
class ReconstructionHead(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, bottleneck_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
    
    def forward(self, hidden_states):
        # 编码
        bottleneck = self.encoder(hidden_states)
        # 解码
        reconstructed = self.decoder(bottleneck)
        # 计算重构损失
        reconstruction_loss = F.mse_loss(
            reconstructed, hidden_states, reduction='none'
        ).mean(dim=1)
        
        return {
            'bottleneck': bottleneck,
            'reconstructed': reconstructed,
            'loss': reconstruction_loss
        }
```

#### 2. 动态融合算法

系统采用动态权重策略融合多种检测方法的结果，提高整体检测性能：

```python
class AnomalyScoreEnsemble(nn.Module):
    def __init__(self, num_detectors=4, fusion_method='dynamic_weight'):
        super().__init__()
        self.fusion_method = fusion_method
        
        # 初始化检测器权重
        if fusion_method == 'dynamic_weight':
            # 使用可学习的权重
            self.weights = nn.Parameter(torch.ones(num_detectors) / num_detectors)
        else:
            # 固定权重
            self.register_buffer('weights', torch.ones(num_detectors) / num_detectors)
    
    def forward(self, detector_scores):
        # 融合多个检测器的分数
        if self.fusion_method == 'max':
            # 最大值融合
            return torch.max(detector_scores, dim=1)[0]
        elif self.fusion_method == 'average':
            # 均值融合
            return torch.mean(detector_scores, dim=1)
        elif self.fusion_method == 'dynamic_weight':
            # 动态权重融合
            weights = F.softmax(self.weights, dim=0)
            return torch.sum(detector_scores * weights.unsqueeze(0), dim=1)
```

#### 3. MLM与对比学习结合的训练算法

系统结合掩码语言模型(MLM)和对比学习两种训练目标，同时学习日志的语义和行为模式：

```python
def compute_loss(self, model, inputs, return_outputs=False):
    # 处理MLM部分
    mlm_labels = inputs.pop('labels', None)
    
    # 处理对比学习部分
    positive_pairs = inputs.pop('positive_pairs', None)
    augment_batch = 'augment' in inputs
    
    # 正向传播，获取输出
    outputs = model(**inputs, 
                    labels=mlm_labels,
                    positive_pairs=positive_pairs,
                    augment_batch=augment_batch,
                    tokenizer=self.tokenizer,
                    training_phase=True)
    
    # 计算总损失
    loss = outputs.loss
    
    return (loss, outputs) if return_outputs else loss
```

#### 4. KNN增强的实时检测算法

系统使用KNN记忆库增强检测能力，为模型提供"记忆"能力：

```python
def score_single_log(self, log_text, use_knn=None):
    # 确定是否使用KNN
    should_use_knn = self.use_knn if use_knn is None else use_knn
    
    if not should_use_knn or self.knn_model is None:
        # 不使用KNN，直接使用detector的detect方法
        result = self.detector.detect(log_text, threshold=self.threshold)
        return result['score']
    else:
        # 使用KNN增强
        # 将单条日志转换为token
        log_tokens = self.log_window.log_tokenizer.tokenize(log_text)
        
        # 执行推理获取embedding和基本分数
        with torch.no_grad():
            outputs = self.model(
                input_ids=log_tokens['input_ids'].to(self.device),
                attention_mask=log_tokens['attention_mask'].to(self.device)
            )
            base_score = outputs['anomaly_score'].item()
            cls_embedding = outputs['cls_embedding'].cpu().numpy()
        
        # 计算KNN分数
        knn_scores = self._compute_knn_scores([cls_embedding])
        knn_score = knn_scores[0]
        
        # 融合两种分数（简单平均）
        combined_score = (base_score + knn_score) / 2
        
        return combined_score
```

### 核心代码结构与关键实现说明

#### 1. 项目结构

```
NLP-SecLogAI/
├── ai_detect/                  # AI检测核心模块
│   ├── core/                   # 核心算法实现
│   │   ├── supervised_evaluator.py    # 评估工具
│   │   ├── data_utils.py              # 数据处理工具
│   │   └── visualization.py           # 可视化工具
│   ├── train.py                # 模型训练脚本
│   ├── evaluate.py             # 模型评估脚本
│   ├── ai_detect_cli.py        # 命令行工具
│   └── requirements.txt        # 依赖列表
├── app/                        # 应用服务模块
│   ├── ai_models/              # AI模型实现
│   │   ├── tinylogbert.py      # 轻量级BERT模型
│   │   ├── anomaly_detector.py # 异常检测器
│   │   ├── log_window.py       # 日志窗口处理
│   │   └── log_tokenizer.py    # 日志标记化
│   ├── services/               # 服务层实现
│   │   └── anomaly_score.py    # 异常评分服务
│   ├── routes/                 # API接口路由
│   │   └── ai_routes.py        # AI检测接口
│   └── docs/                   # 文档说明
│       └── memory_bank/        # 记忆库文档
└── docs/                       # 项目文档
```

#### 2. TinyLogBERT模型实现

轻量级BERT模型是系统的核心组件，基于HuggingFace的BertForMaskedLM扩展而来：

```python
class TinyLogBERT(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        
        # 添加异常检测头
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 添加重构头
        self.reconstruction_head = ReconstructionHead(
            hidden_size=config.hidden_size,
            bottleneck_size=64
        )
        
        # 特征队列，用于对比学习和KNN
        self.register_buffer('queue', torch.randn(config.hidden_size, 2048))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
        # 初始化无监督检测方法
        self.anomaly_methods = {
            'knn': UnsupervisedAnomalyDetector(method='knn'),
            'cluster': UnsupervisedAnomalyDetector(method='cluster'),
            'lof': UnsupervisedAnomalyDetector(method='lof'),
            'iforest': UnsupervisedAnomalyDetector(method='iforest'),
            'reconstruction': None  # 通过重构头实现
        }
        
        # 异常评分融合器
        self.anomaly_ensemble = AnomalyScoreEnsemble(
            num_detectors=len(self.anomaly_methods) + 1,  # +1 是指基本得分
            fusion_method='dynamic_weight'
        )
```

前向传播方法实现了MLM、对比学习和异常检测的多任务目标：

```python
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    positive_pairs=None,  # 用于对比学习的正样本对
    augment_batch=False,  # 是否对批次进行数据增强
    tokenizer=None,       # 用于数据增强的tokenizer
    training_phase=True,  # 是否处于训练阶段
    update_memory=True,   # 是否更新特征内存
):
    # 调用父类BertForMaskedLM的forward方法
    mlm_outputs = super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    
    # 获取[CLS]令牌的隐藏状态，用于异常检测
    sequence_output = mlm_outputs.hidden_states[-1]
    cls_embedding = sequence_output[:, 0]
    
    # 异常检测头：生成异常分数
    anomaly_score = self.anomaly_head(cls_embedding).squeeze(-1)
    
    # 重构头：计算重构损失
    recon_outputs = self.reconstruction_head(cls_embedding)
    recon_loss = recon_outputs['loss']
    
    # 计算多种无监督异常检测方法的评分
    detector_scores = []
    
    # 基本分数(从异常检测头)
    detector_scores.append(anomaly_score)
    
    # 重构损失作为异常分数
    normalized_recon_loss = recon_loss / (torch.max(recon_loss) + 1e-10)
    detector_scores.append(normalized_recon_loss)
    
    # 如果不是训练阶段，添加无监督检测器的分数
    if not training_phase:
        with torch.no_grad():
            features = cls_embedding.detach().cpu().numpy()
            
            for method_name, detector in self.anomaly_methods.items():
                if detector is not None and method_name != 'reconstruction':
                    # 获取检测器评分
                    method_scores = detector.get_anomaly_score(features)
                    method_scores = torch.tensor(
                        method_scores, device=cls_embedding.device
                    )
                    detector_scores.append(method_scores)
    
    # 融合多种异常检测方法的评分
    detector_scores = torch.stack(detector_scores, dim=1)
    ensemble_score = self.anomaly_ensemble(detector_scores)
    
    # 对比学习损失计算（仅在训练阶段）
    contrastive_loss = None
    if training_phase and positive_pairs is not None:
        # 计算对比损失
        # ...对比学习实现代码
    
    # 更新特征队列
    if update_memory and not training_phase:
        self._dequeue_and_enqueue(cls_embedding)
    
    # 整合各损失和输出
    total_loss = mlm_outputs.loss  # MLM损失
    if contrastive_loss is not None:
        total_loss = total_loss + contrastive_loss
    
    return {
        'loss': total_loss,
        'mlm_loss': mlm_outputs.loss,
        'contrastive_loss': contrastive_loss,
        'recon_loss': torch.mean(recon_loss),
        'logits': mlm_outputs.logits,
        'anomaly_score': ensemble_score if not training_phase else anomaly_score,
        'cls_embedding': cls_embedding,
        'bottleneck': recon_outputs['bottleneck'],
        'hidden_states': mlm_outputs.hidden_states,
        'attentions': mlm_outputs.attentions,
    }
```

#### 3. 异常检测服务实现

异常评分服务为系统提供统一的评分接口：

```python
class AnomalyScoreService:
    def __init__(self, model_dir=None, window_size=10, tokenizer_name='prajjwal1/bert-mini', detection_method='ensemble'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.window_size = window_size
        self.threshold = THRESHOLD
        self.batch_size = BATCH_SIZE
        self.use_knn = USE_KNN
        
        # 初始化底层的异常检测器
        self.detector = AnomalyDetector(
            model_dir=model_dir,
            window_size=window_size,
            tokenizer_name=tokenizer_name,
            detection_method=detection_method
        )
        
        self.log_window = LogWindow(tokenizer_name=tokenizer_name, window_size=window_size)
        
        # KNN相关属性
        self.embeddings_bank = None
        self.knn_model = None
    
    def score_single_log(self, log_text, use_knn=None):
        # 单条日志评分实现
        # ...代码已在上面的算法部分展示
    
    def score_log_sequence(self, log_list, window_type='fixed', stride=1, use_knn=None):
        # 日志序列评分实现
        # ...日志序列处理与评分实现
    
    def build_embedding_bank(self, normal_log_windows):
        # 构建特征记忆库
        # ...记忆库构建实现
```

#### 4. API接口实现

系统提供了RESTful API接口，方便与其他系统集成：

```python
@ai_bp.route('/detect', methods=['POST'])
def detect_anomaly():
    try:
        # 获取请求数据
        data = request.get_json()
        
        logs = data['logs']
        window_type = data.get('window_type', 'sliding')
        stride = data.get('stride', 1)
        threshold = data.get('threshold', 0.5)
        use_knn = data.get('use_knn', None)
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        # 检测日志
        if len(logs) == 1:
            # 单条日志
            score = anomaly_service.score_single_log(logs[0], use_knn=use_knn)
            is_anomaly = anomaly_service.is_anomaly(score)
            
            result = {
                "log": logs[0],
                "score": float(score),
                "threshold": float(threshold),
                "is_anomaly": bool(is_anomaly),
                "knn_used": use_knn if use_knn is not None else (anomaly_service.use_knn and anomaly_service.knn_model is not None)
            }
        else:
            # 日志序列
            scores_result = anomaly_service.score_log_sequence(
                logs, window_type=window_type, stride=stride, use_knn=use_knn
            )
            
            scores, avg_score, max_score = scores_result
            
            # 根据阈值识别异常窗口
            windows = []
            for i, score in enumerate(scores):
                # 窗口信息处理
                # ...窗口处理实现
            
            # 找出异常窗口
            anomaly_windows = [w for w in windows if w["is_anomaly"]]
            
            result = {
                "num_windows": len(windows),
                "avg_score": float(avg_score),
                "max_score": float(max_score),
                "num_anomaly_windows": len(anomaly_windows),
                "anomaly_ratio": float(len(anomaly_windows) / len(windows)) if windows else 0.0,
                "windows": windows,
                "knn_used": use_knn if use_knn is not None else (anomaly_service.use_knn and anomaly_service.knn_model is not None)
            }
        
        # 返回结果
        return jsonify({"result": result})
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500
```

## 接口文档

### 接口路径

系统提供以下REST API接口供外部系统集成：

| 接口路径 | 功能描述 |
|---------|---------|
| `/ai/score_log` | 对单条日志进行异常评分 |
| `/ai/score_log_sequence` | 对日志序列进行异常评分 |
| `/ai/detect` | 使用异常检测器检测日志 |
| `/ai/model/status` | 获取模型状态 |
| `/ai/model/threshold` | 设置异常阈值 |
| `/ai/knn/build` | 构建KNN嵌入库 |
| `/ai/knn/status` | 获取或设置KNN增强状态 |
| `/ai/ui` | AI异常检测Web界面 |

### 请求方法

#### 1. `/ai/score_log` 接口

**方法**: POST

**功能**: 对单条日志进行异常评分，返回异常分数和判定结果

**权限要求**: 无特殊权限要求

#### 2. `/ai/score_log_sequence` 接口

**方法**: POST

**功能**: 对日志序列进行异常评分，返回窗口评分和异常窗口

**权限要求**: 无特殊权限要求

#### 3. `/ai/detect` 接口

**方法**: POST

**功能**: 使用异常检测器检测日志，支持单条和序列日志检测

**权限要求**: 无特殊权限要求

#### 4. `/ai/model/status` 接口

**方法**: GET

**功能**: 获取模型状态，包括加载状态、设备信息和配置参数

**权限要求**: 无特殊权限要求

#### 5. `/ai/model/threshold` 接口

**方法**: POST

**功能**: 设置异常判定阈值

**权限要求**: 可配置为需要管理员权限

#### 6. `/ai/knn/build` 接口

**方法**: POST

**功能**: 构建KNN嵌入库，用于增强检测能力

**权限要求**: 可配置为需要管理员权限

#### 7. `/ai/knn/status` 接口

**方法**: GET/POST

**功能**: 获取或设置KNN增强状态

**权限要求**: GET不需要权限，POST可配置为需要管理员权限

#### 8. `/ai/ui` 接口

**方法**: GET

**功能**: 返回AI异常检测Web界面

**权限要求**: 无特殊权限要求

### 请求参数

#### 1. `/ai/score_log` 接口

**请求体格式**: JSON

```json
{
  "log": "日志文本内容",
  "use_knn": true,  // 可选，是否使用KNN增强
  "threshold": 0.5  // 可选，异常判定阈值
}
```

#### 2. `/ai/score_log_sequence` 接口

**请求体格式**: JSON

```json
{
  "logs": ["日志1", "日志2", "日志3", ...],
  "window_type": "fixed",  // 可选，窗口类型，默认"fixed"
  "stride": 1,             // 可选，滑动窗口步长，默认1
  "use_knn": true,         // 可选，是否使用KNN增强
  "threshold": 0.5         // 可选，异常判定阈值
}
```

#### 3. `/ai/detect` 接口

**请求体格式**: JSON

```json
{
  "logs": ["日志1", "日志2", "日志3", ...],
  "window_type": "sliding", // 可选，窗口类型，默认"sliding"
  "stride": 1,              // 可选，滑动窗口步长，默认1
  "threshold": 0.5,         // 可选，异常判定阈值，默认0.5
  "use_knn": true           // 可选，是否使用KNN增强
}
```

#### 4. `/ai/model/status` 接口

无请求参数。

#### 5. `/ai/model/threshold` 接口

**请求体格式**: JSON

```json
{
  "threshold": 0.5  // 新的异常判定阈值，0-1之间
}
```

#### 6. `/ai/knn/build` 接口

**请求体格式**: JSON

```json
{
  "normal_logs": ["正常日志1", "正常日志2", ...]  // 用于训练KNN的正常日志样本
}
```

#### 7. `/ai/knn/status` 接口

**GET请求**: 无参数

**POST请求体格式**: JSON

```json
{
  "enabled": true  // 是否启用KNN增强
}
```

### 返回值结构

#### 1. `/ai/score_log` 接口

**成功响应**: 状态码 200

```json
{
  "log": "日志文本内容",
  "score": 0.82,           // 异常分数，0-1之间
  "threshold": 0.5,        // 当前使用的阈值
  "is_anomaly": true,      // 是否判定为异常
  "knn_used": true         // 是否使用了KNN增强
}
```

**错误响应**: 状态码 400/500

```json
{
  "error": "错误信息"
}
```

#### 2. `/ai/score_log_sequence` 接口

**成功响应**: 状态码 200

```json
{
  "scores": [0.1, 0.2, 0.9, 0.3, ...],  // 各窗口的异常分数
  "avg_score": 0.35,                    // 平均异常分数
  "max_score": 0.9,                     // 最大异常分数
  "threshold": 0.5,                     // 当前使用的阈值
  "num_windows": 5,                     // 窗口总数
  "anomaly_windows": [2],               // 异常窗口索引
  "num_anomaly_windows": 1,             // 异常窗口数量
  "knn_used": true                      // 是否使用了KNN增强
}
```

**错误响应**: 状态码 400/500

```json
{
  "error": "错误信息"
}
```

#### 3. `/ai/detect` 接口

**成功响应 (单条日志)**: 状态码 200

```json
{
  "result": {
    "log": "日志文本内容",
    "score": 0.82,
    "threshold": 0.5,
    "is_anomaly": true,
    "knn_used": true
  }
}
```

**成功响应 (日志序列)**: 状态码 200

```json
{
  "result": {
    "num_windows": 10,
    "avg_score": 0.35,
    "max_score": 0.9,
    "num_anomaly_windows": 2,
    "anomaly_ratio": 0.2,
    "windows": [
      {
        "window_idx": 0,
        "start_idx": 0,
        "end_idx": 5,
        "logs": ["日志1", "日志2", ...],
        "score": 0.2,
        "is_anomaly": false
      },
      {
        "window_idx": 1,
        "start_idx": 5,
        "end_idx": 10,
        "logs": ["日志6", "日志7", ...],
        "score": 0.85,
        "is_anomaly": true
      },
      // 更多窗口...
    ],
    "knn_used": true
  }
}
```

**错误响应**: 状态码 400/500

```json
{
  "error": "错误信息"
}
```

#### 4. `/ai/model/status` 接口

**成功响应**: 状态码 200

```json
{
  "model_loaded": true,
  "device": "cuda:0",
  "threshold": 0.5,
  "window_size": 10,
  "batch_size": 32,
  "knn_enabled": true
}
```

**错误响应**: 状态码 500

```json
{
  "model_loaded": false,
  "error": "错误信息"
}
```

#### 5. `/ai/model/threshold` 接口

**成功响应**: 状态码 200

```json
{
  "success": true,
  "threshold": 0.5  // 更新后的阈值
}
```

**错误响应**: 状态码 400/500

```json
{
  "error": "错误信息"
}
```

#### 6. `/ai/knn/build` 接口

**成功响应**: 状态码 200

```json
{
  "success": true,
  "num_embeddings": 1000,
  "message": "成功构建KNN嵌入库"
}
```

**错误响应**: 状态码 400/500

```json
{
  "success": false,
  "message": "KNN嵌入库构建失败",
  "error": "错误信息"
}
```

#### 7. `/ai/knn/status` 接口

**GET成功响应**: 状态码 200

```json
{
  "knn_enabled": true,
  "knn_available": true,
  "num_embeddings": 1000
}
```

**POST成功响应**: 状态码 200

```json
{
  "success": true,
  "knn_enabled": true,
  "knn_available": true,
  "num_embeddings": 1000
}
```

**错误响应**: 状态码 400/500

```json
{
  "error": "错误信息"
}
```

### 状态码说明

| 状态码 | 说明 |
|-------|------|
| 200 | 请求成功 |
| 400 | 请求参数错误或格式不正确 |
| 401 | 未授权访问（如需要管理员权限的接口） |
| 404 | 请求的资源不存在 |
| 500 | 服务器内部错误 |

#### 
