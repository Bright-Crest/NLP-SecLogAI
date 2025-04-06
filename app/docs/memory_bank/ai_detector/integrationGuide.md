# AI日志异常检测模块 - 集成指南

## 概述

本文档提供了将AI日志异常检测模块集成到现有系统或应用中的详细指南。AI异常检测模块提供了强大的API接口，可以轻松地与各种安全系统和日志分析平台集成。

## 集成方式

### 1. REST API集成

AI异常检测模块提供了RESTful API接口，这是最常用和推荐的集成方式。

#### 基本信息

- **基础URL**: `http://<server-address>:<port>/api/v1/anomaly`
- **认证方式**: API密钥或JWT令牌
- **请求格式**: JSON
- **响应格式**: JSON

#### 主要接口

| 接口 | 方法 | 描述 | 参数 |
|------|------|------|------|
| `/detect` | POST | 检测单条日志异常 | 日志文本、上下文标识 |
| `/batch_detect` | POST | 批量检测多条日志异常 | 日志数组、配置参数 |
| `/feedback` | POST | 提交用户反馈 | 检测ID、反馈类型 |
| `/threshold` | GET/PUT | 获取/设置检测阈值 | 阈值值、应用范围 |
| `/stats` | GET | 获取检测统计信息 | 时间范围 |

#### 示例调用

##### 单条日志检测
```http
POST /api/v1/anomaly/detect HTTP/1.1
Host: example.com
Authorization: Bearer <token>
Content-Type: application/json

{
  "log_text": "Failed password for invalid user admin from 192.168.1.100 port 54216 ssh2",
  "log_type": "ssh",
  "context_id": "session-123"
}
```

##### 响应示例
```json
{
  "detection_id": "d7f9a1b2-c3d4-e5f6-7a8b-9c0d1e2f3a4b",
  "timestamp": "2023-04-12T15:23:45Z",
  "anomaly_score": 0.87,
  "is_anomaly": true,
  "confidence": 92,
  "explanation": {
    "factors": [
      {"factor": "IP地址异常", "contribution": 0.65},
      {"factor": "用户名异常", "contribution": 0.22}
    ],
    "similar_patterns": [
      {"id": "pattern-456", "similarity": 0.89}
    ]
  },
  "recommended_action": "调查此IP地址的其他活动"
}
```

##### 批量检测
```http
POST /api/v1/anomaly/batch_detect HTTP/1.1
Host: example.com
Authorization: Bearer <token>
Content-Type: application/json

{
  "logs": [
    {"text": "Failed password for invalid user admin from 192.168.1.100 port 54216 ssh2", "type": "ssh"},
    {"text": "Accepted password for user1 from 10.0.0.15 port 22345 ssh2", "type": "ssh"},
    {"text": "Connection closed by 192.168.1.100 port 54216", "type": "ssh"}
  ],
  "options": {
    "context_aware": true,
    "threshold": 0.75,
    "max_results": 100
  }
}
```

### 2. 库集成

对于Python应用，可以直接使用AI异常检测模块的Python客户端库。

#### 安装

```bash
pip install nlp-seclogai-anomaly-client
```

#### 基本用法

```python
from seclogai.anomaly import AnomalyDetector

# 初始化检测器
detector = AnomalyDetector(api_key="your_api_key")

# 单条日志检测
result = detector.detect("Failed password for invalid user admin from 192.168.1.100 port 54216 ssh2")
if result.is_anomaly:
    print(f"发现异常! 分数: {result.score}")
    print(f"原因: {result.explanation}")

# 批量检测
logs = [
    "Failed password for invalid user admin from 192.168.1.100 port 54216 ssh2",
    "Accepted password for user1 from 10.0.0.15 port 22345 ssh2",
    "Connection closed by 192.168.1.100 port 54216"
]
results = detector.batch_detect(logs)
for i, result in enumerate(results):
    if result.is_anomaly:
        print(f"日志 #{i+1} 是异常的，分数: {result.score}")
```

### 3. 消息队列集成

对于需要高吞吐量的应用场景，可以通过消息队列集成。

#### 支持的消息队列系统

- Apache Kafka
- RabbitMQ
- Redis Streams

#### Kafka集成示例

##### 生产者配置

```json
{
  "bootstrap.servers": "kafka:9092",
  "input.topic": "raw_logs",
  "output.topic": "anomaly_results",
  "consumer.group": "anomaly_detector",
  "threshold": 0.8,
  "batch.size": 100
}
```

##### 使用Docker Compose部署

```yaml
version: '3'
services:
  anomaly-detector:
    image: nlp-seclogai/anomaly-detector:latest
    volumes:
      - ./config:/app/config
    environment:
      - KAFKA_BROKER=kafka:9092
      - INPUT_TOPIC=raw_logs
      - OUTPUT_TOPIC=anomaly_results
      - MODEL_PATH=/app/models/tinylogbert
      - THRESHOLD=0.8
    depends_on:
      - kafka
  
  kafka:
    image: confluentinc/cp-kafka:latest
    ports:
      - "9092:9092"
    environment:
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1
```

## 配置参数

### 核心配置参数

| 参数 | 描述 | 默认值 | 可选值 |
|------|------|-------|-------|
| `model.path` | 模型文件路径 | `/app/models/tinylogbert` | 任何有效路径 |
| `threshold.default` | 默认异常检测阈值 | `0.7` | 0.0-1.0 |
| `threshold.adaptive` | 是否启用自适应阈值 | `true` | true/false |
| `window.size` | 日志窗口大小 | `10` | 1-100 |
| `cache.enabled` | 是否启用向量缓存 | `true` | true/false |
| `cache.size` | 缓存大小(条目数) | `1000` | 100-10000 |
| `detectors` | 启用的检测器列表 | `["iforest", "knn"]` | iforest,knn,autoencoder,lof |
| `api.port` | API服务端口 | `5000` | 1024-65535 |
| `log.level` | 日志级别 | `info` | debug,info,warning,error |

### 示例配置文件

```yaml
# config.yaml
model:
  path: /app/models/tinylogbert
  version: 1.2.0
  quantized: true

threshold:
  default: 0.75
  adaptive: true
  min: 0.5
  max: 0.95

window:
  size: 15
  step: 5
  time_based: false

cache:
  enabled: true
  size: 2000
  ttl: 3600

detectors:
  - name: iforest
    weight: 0.4
    params:
      n_estimators: 100
      contamination: 0.1
  - name: knn
    weight: 0.4
    params:
      n_neighbors: 5
  - name: autoencoder
    weight: 0.2
    params:
      hidden_dim: 32

api:
  port: 5000
  workers: 4
  timeout: 30
  rate_limit: 100

log:
  level: info
  file: /var/log/anomaly_detector.log
  rotation: daily
```

## 安全注意事项

### 认证与授权

AI异常检测模块API支持多种认证方式：

1. **API密钥认证**：最简单的集成方式，适合内部系统集成
   ```http
   GET /api/v1/anomaly/stats HTTP/1.1
   Host: example.com
   X-API-Key: your_api_key_here
   ```

2. **JWT令牌认证**：提供更高级的安全性和用户身份识别
   ```http
   POST /api/v1/anomaly/detect HTTP/1.1
   Host: example.com
   Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

3. **基于IP的访问控制**：可以配置白名单IP地址，限制API访问

### 数据安全

- 所有API通信应使用HTTPS加密
- 敏感日志数据可以在传输前进行脱敏处理
- 支持对敏感字段进行自动脱敏(如密码、令牌等)

## 性能考量

### 规模与容量

| 部署规模 | 日志量 | 推荐配置 | 支持并发度 |
|---------|-------|---------|----------|
| 小型 | < 1GB/天 | 2核4GB | 10 |
| 中型 | 1-10GB/天 | 4核8GB | 50 |
| 大型 | 10-100GB/天 | 8核16GB | 200 |
| 超大型 | > 100GB/天 | 16核32GB+ | 500+ |

### 性能优化技巧

1. **启用缓存**：对于重复出现的日志模式，启用向量缓存可显著提高性能
2. **批量处理**：尽可能使用批量API而非单条处理
3. **调整窗口大小**：较小的窗口大小可以提高处理速度，但可能影响上下文感知能力
4. **模型量化**：使用量化版本的模型可以提高推理速度
5. **分布式部署**：对于超大规模场景，考虑水平扩展部署多个实例

## 集成最佳实践

1. **渐进式集成**：先在非关键系统上集成并验证效果，再扩展到核心系统
2. **日志抽样**：对于超大规模系统，可以考虑抽样分析而非全量处理
3. **配置阶段性阈值**：根据时间段和系统特性设置不同的阈值
4. **结合规则引擎**：将AI检测结果与传统规则引擎结合使用，取长补短
5. **定期反馈**：实现用户反馈机制，帮助系统不断学习和改进
6. **监控与告警**：设置对检测系统本身的监控和告警机制
7. **定期更新模型**：安排定期的模型更新计划，适应不断变化的威胁环境

## 故障排除

### 常见问题及解决方案

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| API调用超时 | 处理负载过高 | 增加超时设置，启用批处理，检查系统资源 |
| 异常评分不准确 | 模型与实际环境不匹配 | 提供反馈，使用本地数据重新训练模型 |
| 高误报率 | 阈值设置过低 | 调整阈值，提供反馈优化模型 |
| 批处理性能低 | 批量大小不合适 | 调整批处理大小，优化内存配置 |
| 模型加载失败 | 模型文件损坏或不兼容 | 检查模型文件完整性，确认版本兼容性 |

### 诊断工具

1. **健康检查API**：用于验证服务状态
   ```
   GET /api/v1/health
   ```

2. **性能分析API**：提供系统性能指标
   ```
   GET /api/v1/performance
   ```

3. **日志设置**：增加日志详细程度进行调试
   ```
   PUT /api/v1/settings/log
   {"level": "debug"}
   ```

## 实际集成案例

### 案例1：与SIEM系统集成

```mermaid
graph LR
    A[日志源] --> B[SIEM收集器]
    B --> C[SIEM平台]
    C --> D[AI异常检测模块]
    D --> E[强化告警]
    E --> C
```

**实现步骤**：
1. 配置SIEM平台将收集的日志通过API发送到异常检测模块
2. 异常检测模块分析后返回评分和解释
3. SIEM平台根据异常评分调整告警优先级

### 案例2：与日志聚合系统集成

```mermaid
graph LR
    A[应用服务器] --> B[日志聚合器]
    B --> C[Elasticsearch]
    B --> D[AI异常检测模块]
    D --> E[异常检测结果]
    E --> C
    C --> F[Kibana仪表盘]
```

**实现步骤**：
1. 配置日志聚合器(如Logstash)将日志同时发送到Elasticsearch和异常检测API
2. 异常检测结果写回Elasticsearch中，关联原始日志
3. 在Kibana中创建仪表盘显示异常检测结果和趋势

## 支持资源

- **技术支持**：support@seclogai.com
- **文档网站**：https://docs.seclogai.com/anomaly-detector/
- **API文档**：https://api.seclogai.com/swagger/
- **GitHub仓库**：https://github.com/seclogai/anomaly-detector
- **问题跟踪**：https://github.com/seclogai/anomaly-detector/issues

---

*文档版本：1.0.0 - 最后更新：2023-04-12*