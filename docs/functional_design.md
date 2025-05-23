# 三、功能设计

## 3.1 日志收集与解析模块

### 功能描述

日志收集与解析模块负责从多种来源获取安全日志，将不同格式的日志标准化为系统可处理的统一格式。模块设计支持多种日志格式，并提供可扩展的接口以便未来添加新的日志类型支持。

### 主要功能点

1. **多源日志收集**
   - 文件导入：支持批量上传日志文件
   - Syslog接收：作为Syslog服务器接收网络设备日志
   - API接口：允许外部系统通过API推送日志
   - 代理采集：可部署轻量级代理收集远程系统日志

2. **日志格式解析**
   - SSH日志解析
   - Web服务器日志解析
   - 防火墙日志解析
   - 数据库日志解析
   - HDFS和Linux系统日志解析

3. **日志标准化**
   - 时间标准化：转换各种时间格式为统一格式
   - 字段提取：从非结构化日志中提取关键字段
   - 格式转换：将解析结果转换为系统内部统一结构

4. **元数据增强**
   - IP地理位置识别
   - 主机名解析
   - 事件类型分类
   - 日志严重性评估

### 流程图

```
           ┌─────────────┐
           │  日志源     │
           └──────┬──────┘
                  │
                  ▼
┌─────────────────────────────────┐
│        日志收集接口             │
│  ┌──────┐ ┌──────┐ ┌──────┐    │
│  │文件  │ │Syslog│ │ API  │    │
│  └──────┘ └──────┘ └──────┘    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│        日志格式检测             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│        日志解析策略选择         │
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        │             │
┌───────▼───────┐    ┌▼─────────────┐
│ SSH日志解析器 │    │Web日志解析器 │ ...
└───────┬───────┘    └┬─────────────┘
        │             │
        └──────┬──────┘
               │
               ▼
┌─────────────────────────────────┐
│       日志标准化处理            │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│       元数据增强                │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│     输出标准化日志记录          │
└─────────────────────────────────┘
```

### 模块接口

#### 输入接口
- **文件上传API**：`POST /logs/upload`
- **日志流接收接口**：`POST /logs/stream`
- **Syslog接收器**：UDP端口514

#### 输出接口
- **标准化日志**：结构化JSON格式
- **解析状态反馈**：成功/失败统计，错误日志

### 关键技术实现
- 采用**策略模式**实现不同日志类型的解析
- 使用**正则表达式**进行日志格式匹配和字段提取
- 采用**工厂模式**动态创建适合的解析器
- 实现**观察者模式**通知下游模块处理新日志

## 3.2 NLP处理引擎

### 功能描述

NLP处理引擎负责对标准化后的日志内容进行自然语言处理分析，提取关键信息和语义理解，为异常检测提供基础。模块设计支持增量学习，能够随着系统使用不断提高分析准确性。

### 主要功能点

1. **文本预处理**
   - 文本清洗：去除特殊字符、格式化
   - 分词和标记化：将日志拆分为有意义的单元
   - 停用词过滤：移除分析中无意义的常见词

2. **实体识别**
   - IP地址、端口、用户名识别
   - 系统路径和文件名识别
   - 指令和参数识别
   - 错误代码和状态码识别

3. **特征提取**
   - 关键词提取和频率分析
   - TF-IDF特征计算
   - 语义向量生成
   - 异常特征标记

4. **语义分析**
   - 日志事件类型分类
   - 日志严重性评估
   - 行为模式识别
   - 关联关系发现

### 流程图

```
┌─────────────────────┐
│  标准化日志输入     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    文本预处理       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     分词/标记化     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     实体识别        │
└──────────┬──────────┘
           │
      ┌────┴─────┐
      │          │
      ▼          ▼
┌─────────┐ ┌─────────┐
│特征提取 │ │语义分析 │
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────────┐
│  NLP分析结果输出    │
└─────────────────────┘
```

### 模块接口

#### 输入接口
- **日志分析API**：`POST /nlp/analyze`
- **批量分析接口**：`POST /nlp/batch-analyze`
- **内部处理接口**：供日志处理管道调用

#### 输出接口
- **结构化NLP分析结果**：JSON格式
- **实体识别结果**：识别出的命名实体及其类型
- **特征向量**：用于后续异常检测的特征

### 关键技术实现
- 基于**spaCy/NLTK**的自然语言处理管道
- 采用**词向量模型**（Word2Vec/FastText）提取语义特征
- 使用**命名实体识别**（NER）提取结构化信息
- 实现**增量学习机制**持续优化模型准确性

## 3.3 异常检测模块

### 功能描述

异常检测模块是系统的核心组件，负责基于规则检测和AI检测两种方式识别安全日志中的异常和潜在威胁。模块采用双引擎设计，规则引擎负责已知威胁模式检测，AI引擎负责未知异常和零日威胁检测。

### 主要功能点

1. **规则检测引擎**
   - SSH暴力破解检测
   - Web攻击识别(SQL注入、XSS等)
   - 防火墙异常行为分析
   - 数据库安全事件监测
   - 系统文件操作异常检测

2. **AI异常检测**
   - 基于统计模型的异常检测
   - 基于机器学习的行为建模
   - 基于NLP特征的聚类分析
   - 用户行为异常识别

3. **威胁关联分析**
   - 多源日志关联
   - 攻击链识别
   - 事件时序分析
   - 威胁情报整合

4. **告警管理**
   - 告警分级与评分
   - 告警合并与去重
   - 告警通知与分发
   - 误报管理与优化

### 流程图

```
┌────────────────┐   ┌────────────────┐
│  标准化日志    │   │  NLP分析结果   │
└───────┬────────┘   └────────┬───────┘
        │                     │
        └──────────┬──────────┘
                   │
         ┌─────────▼─────────┐
         │  异常检测分发器   │
         └─────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
┌───────▼────────┐   ┌────────▼───────┐
│  规则检测引擎  │   │  AI检测引擎    │
└───────┬────────┘   └────────┬───────┘
        │                     │
        └──────────┬──────────┘
                   │
         ┌─────────▼─────────┐
         │   结果合并处理    │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  威胁关联分析     │
         └─────────┬─────────┘
                   │
                   ▼
┌──────────────────────────────┐
│           告警生成           │
└──────────────────────────────┘
```

### 模块接口

#### 输入接口
- **异常检测API**：`POST /anomalies/detect`
- **特定日志类型检测**：`GET /anomalies/detect/{log_type}`
- **批量检测接口**：`POST /anomalies/batch-detect`

#### 输出接口
- **异常检测结果**：包含异常类型、严重性、证据和建议
- **告警通知**：生成系统告警和可选的外部通知
- **威胁情报更新**：用于更新规则库的反馈

### 关键技术实现
- 基于**SQL查询**实现灵活的规则检测
- 使用**异常检测算法**（Isolation Forest、LOF等）实现AI检测
- 采用**动态阈值**技术减少误报
- 实现**规则自学习**机制从AI检测结果生成新规则

## 3.4 Web界面模块

### 功能描述

Web界面模块提供直观的用户交互界面，展示安全日志分析结果、异常告警和系统状态。界面设计符合现代UI/UX标准，提供仪表盘、详细报告和配置管理功能。

### 主要功能点

1. **安全仪表盘**
   - 实时安全态势展示
   - 异常统计和趋势图表
   - 重要安全事件通知
   - 系统健康状态监控

2. **日志管理**
   - 日志浏览与搜索
   - 日志过滤和排序
   - 详细日志查看
   - 日志导出功能

3. **告警管理**
   - 告警列表与详情
   - 告警处理工作流
   - 告警优先级管理
   - 历史告警查询

4. **报表生成**
   - 安全态势报告
   - 异常事件报告
   - 定制化报表
   - 导出多种格式（PDF、Excel等）

5. **系统管理**
   - 用户和权限管理
   - 系统配置
   - 规则管理
   - 系统监控与日志

### 界面结构图

```
┌────────────────────────────────────────────────────────────┐
│                         顶部导航栏                         │
├────────────┬────────────┬────────────────┬────────────────┤
│            │            │                │                │
│  仪表盘    │  日志管理  │   告警管理     │   系统管理     │
│            │            │                │                │
├────────────┴────────────┴────────────────┴────────────────┤
│                                                            │
│                                                            │
│                       主内容区域                           │
│                                                            │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                         底部状态栏                         │
└────────────────────────────────────────────────────────────┘
```

### 模块接口

#### 前端接口
- **登录入口**：`/login`
- **主仪表盘**：`/dashboard`
- **日志管理**：`/logs`
- **告警管理**：`/alerts`
- **系统设置**：`/settings`

#### 与后端通信接口
- **认证API**：`POST /api/auth/login`
- **数据获取**：`GET /api/{resource}`
- **数据更新**：`POST/PUT /api/{resource}`
- **WebSocket实时更新**：`ws://host/ws`

### 关键技术实现
- 基于**Vue.js**框架构建响应式前端
- 使用**Element UI**组件库提供统一界面元素
- 采用**ECharts**实现可视化图表和仪表盘
- 实现**响应式设计**支持多种设备访问

## 3.5 API服务层

### 功能描述

API服务层为系统提供统一的RESTful接口，支持前端界面访问和第三方系统集成。API设计遵循REST规范，提供完整的资源操作能力和良好的可扩展性。

### 主要功能点

1. **认证与授权**
   - 用户认证接口
   - 令牌管理
   - 权限控制
   - API密钥管理

2. **日志管理API**
   - 日志上传接口
   - 日志查询接口
   - 日志统计接口
   - 日志导出接口

3. **分析与检测API**
   - NLP分析接口
   - 异常检测接口
   - 威胁情报获取接口
   - 安全评分接口

4. **告警管理API**
   - 告警查询接口
   - 告警处理接口
   - 告警规则配置接口
   - 告警通知配置接口

5. **系统管理API**
   - 用户管理接口
   - 系统配置接口
   - 规则管理接口
   - 系统监控接口

### API结构图

```
/api
  │
  ├── /auth
  │    ├── /login
  │    ├── /logout
  │    └── /refresh
  │
  ├── /logs
  │    ├── /upload
  │    ├── /search
  │    ├── /types
  │    └── /{id}
  │
  ├── /nlp
  │    ├── /analyze
  │    └── /extract
  │
  ├── /anomalies
  │    ├── /detect
  │    ├── /rules
  │    └── /stats
  │
  ├── /alerts
  │    ├── /active
  │    ├── /history
  │    └── /{id}
  │
  └── /system
       ├── /users
       ├── /settings
       └── /status
```

### 接口约定

#### 请求格式
- **内容类型**：`application/json`
- **认证头**：`Authorization: Bearer {token}`
- **分页参数**：`page`、`limit`
- **排序参数**：`sort=field:direction`

#### 响应格式
```json
{
  "status": "success|error",
  "data": { ... },
  "meta": {
    "page": 1,
    "total": 100,
    "limit": 10
  },
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description"
  }
}
```

### 关键技术实现
- 使用**Flask-RESTful**构建标准化API
- 采用**JWT**实现无状态认证
- 实现**API版本控制**支持平滑升级
- 提供**Swagger**自动生成API文档 