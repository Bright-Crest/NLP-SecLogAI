# NL2SQL模块设计文档

## 系统概述

### 总体架构设计

NL2SQL模块采用多层架构设计，通过清晰的职责分离实现系统的可维护性与可扩展性。

#### 架构图

```
┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Web界面/API │────>│ 查询预处理器  │────>│ SQL转换服务     │
└─────────────┘     └───────────────┘     └─────────────────┘
       │                                           │
       │                                           ▼
       │                                  ┌─────────────────┐
       │                                  │ 实体增强器      │
       │                                  └─────────────────┘
       │                                           │
       ▼                                           ▼
┌─────────────┐                          ┌─────────────────┐
│ 响应格式化  │<─────────────────────────│ 数据库查询执行  │
└─────────────┘                          └─────────────────┘
```

#### 架构描述

NL2SQL模块由以下主要组件构成：

1. **Web界面/API层**：提供用户界面和REST API接口，接收自然语言查询请求
2. **查询预处理器**：解析用户输入，识别时间表达式、实体和查询意图
3. **SQL转换服务**：将预处理后的查询转换为SQL语句，支持API调用和模式匹配两种策略
4. **实体增强器**：对生成的SQL进行优化，增强实体匹配能力
5. **数据库查询执行器**：执行生成的SQL语句并获取结果
6. **响应格式化**：将查询结果格式化为JSON响应返回给用户

### 系统边界与模块划分

#### 系统边界

- **输入边界**：自然语言查询文本
- **输出边界**：SQL查询语句和执行结果
- **外部依赖**：OpenRouter API用于NLP处理

#### 模块划分

1. **查询处理模块**
   - 查询预处理
   - 实体识别
   - 意图分析

2. **转换模块**
   - LLM转换策略
   - 模式匹配策略
   - SQL提取与验证

3. **数据访问模块**
   - 数据库连接管理
   - SQL执行
   - 结果处理

4. **接口模块**
   - REST API
   - Web界面
   - 错误处理

### 关键技术选型与理由

| 技术 | 选型 | 理由 |
|------|------|------|
| NLP引擎 | OpenRouter API (Claude/GPT) | 提供统一接口访问多种大型语言模型，无需部署模型，适合轻量级应用 |
| 后端框架 | Flask | 轻量级Web框架，易于集成，适合构建REST API服务 |
| 数据库 | SQLite | 嵌入式数据库，零配置，适合中小规模应用，方便部署和维护 |
| 正则表达式 | Python re模块 | 内置库，用于模式匹配和预处理，性能适中 |
| 数据处理 | Python标准库 | 丰富的内置工具，简化开发 |
| 前端框架 | Bootstrap + jQuery | 广泛使用的UI框架，快速构建响应式界面 |

### 系统运行环境要求

#### 硬件要求
- CPU：1.6GHz双核处理器或更高
- 内存：最低4GB，推荐8GB或更高
- 存储：至少100MB可用空间

#### 操作系统
- Windows 10/11
- MacOS 11.0+
- Linux (Ubuntu 20.04+, CentOS 8+)

#### 软件环境
- Python 3.7+
- SQLite 3.0+
- 网络连接（用于API调用）

#### 中间件
- Web服务器（可选）：Nginx/Apache作为反向代理
- WSGI服务器（可选）：Gunicorn/uWSGI用于生产环境部署

## 五、功能设计

### 模块功能描述

#### 1. 查询预处理器

**功能**：
- 时间表达式识别与解析
- 实体识别（用户名、IP地址等）
- 查询意图初步分析
- 关键词映射与标准化

**主要方法**：
- `_preprocess_query(query_text)`: 预处理查询文本
- `_update_time_range(time_range, value, unit)`: 更新时间范围
- `_handle_special_time_range(time_range, time_expr)`: 处理特殊时间表达式
- `_parse_absolute_time_range(time_range, start, end)`: 解析绝对时间范围
- `_map_status_keyword(text, status_word)`: 映射状态关键词

#### 2. SQL转换服务

**功能**：
- 构建LLM查询提示
- 调用OpenRouter API
- 从API响应中提取SQL
- 提供模式匹配回退机制

**主要方法**：
- `convert(query_text)`: 主转换入口
- `_build_prompt(processed_query, time_range)`: 构建提示
- `_call_openrouter_api(prompt)`: 调用API
- `_extract_sql(text)`: 提取SQL
- `_pattern_matching_fallback(query_text, time_range)`: 回退方法

#### 3. 实体增强器

**功能**：
- 增强实体匹配模式
- 处理复杂查询（如UNION查询）
- 优化SQL性能

**主要方法**：
- `_enhance_entity_matching(sql, processed_query)`: 增强实体匹配
- `_enhance_single_query(sql, processed_query)`: 增强单个查询
- `_generate_multiple_matches(field, value)`: 生成多种匹配条件

#### 4. 数据库查询执行器

**功能**：
- 连接数据库
- 执行SQL查询
- 格式化结果集

**主要方法**：
- `execute_query(sql)`: 执行查询
- `format_results(cursor)`: 格式化结果

#### 5. Web界面与API

**功能**：
- 提供REST API
- 展示Web界面
- 处理请求和响应

**主要方法**：
- `nlp_query()`: 处理NLP查询请求
- `nlp_ui()`: 提供Web界面

### 流程图

#### 主流程

```
┌─────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────────────┐     ┌────────────┐
│ 开始    │────>│ 接收用户查询  │────>│ 查询预处理   │────>│ SQL转换生成  │────>│ 实体增强    │────>│ 执行SQL查询│
└─────────┘     └───────────────┘     └──────────────┘     └──────────────┘     └─────────────┘     └────────────┘
                                                                  │                                        │
                                                                  │                                        │
                                                                  ▼                                        ▼
                                                           ┌──────────────┐                         ┌────────────┐
                                                           │ 转换失败     │                         │ 格式化结果 │
                                                           └──────────────┘                         └────────────┘
                                                                  │                                        │
                                                                  ▼                                        │
                                                           ┌──────────────┐                                │
                                                           │ 使用回退方法 │                                │
                                                           └──────────────┘                                │
                                                                  │                                        │
                                                                  ▼                                        ▼
                                                           ┌──────────────────────────────────────────────────┐
                                                           │ 返回响应                                         │
                                                           └──────────────────────────────────────────────────┘
```

#### 查询预处理流程

```
┌─────────┐     ┌───────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌────────────────┐
│ 开始    │────>│ 初始化时间范围│────>│ 处理时间表达式 │────>│ 识别用户名/IP   │────>│ 识别事件类型   │
└─────────┘     └───────────────┘     └────────────────┘     └─────────────────┘     └────────────────┘
                                                                                             │
                                                                                             │
                                                                                             ▼
┌────────────────────────┐     ┌─────────────────────┐     ┌────────────────────┐     ┌────────────────┐
│ 返回处理后查询和时间范围│<────│ 添加排序和限制条件  │<────│ 处理特定查询类型   │<────│ 识别状态关键词 │
└────────────────────────┘     └─────────────────────┘     └────────────────────┘     └────────────────┘
```

### 接口定义

#### 模块内部接口

1. **NL2SQLConverter 类接口**

```python
class NL2SQLConverter:
    def __init__(self, model_name="openrouter/auto", table_schema=None, max_tokens=500, temperature=0.1):
        # 初始化转换器
        
    def convert(self, query_text: str) -> Dict[str, Any]:
        # 将自然语言查询转换为SQL
        # 返回：包含SQL查询、置信度和原始查询的字典
```

2. **工具函数接口**

```python
def convert_to_sql(query_text: str, table_schema=None) -> str:
    # 简化版转换函数，直接返回SQL字符串
```

#### 对外REST API接口

1. **NLP查询API**
   - 路径: `/nlp/query`
   - 方法: POST
   - 请求体: `{"query": "自然语言查询文本"}`
   - 响应: `{"sql": "生成的SQL", "results": [...], "original_query": "原始查询"}`

2. **NLP UI接口**
   - 路径: `/nlp/ui`
   - 方法: GET
   - 响应: HTML页面

## 六、详细设计

### 数据库设计

系统使用三个主要表存储数据：logs（日志表）、rules（规则表）和anomalies（异常表）。

#### 表结构设计

##### 1. 日志表 (logs)

存储所有安全日志，供规则检测和AI分析。

```sql
CREATE TABLE logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
    source_ip TEXT, 
    event_type TEXT, 
    message TEXT, 
    detected_by TEXT DEFAULT 'none', -- 'rules', 'AI', 'both', 'none'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

##### 2. 规则表 (rules)

存储SQL规则，规则检测模块会执行sql_query进行匹配。

```sql
CREATE TABLE rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    name TEXT UNIQUE NOT NULL, 
    description TEXT, 
    sql_query TEXT NOT NULL, 
    action TEXT CHECK(action IN ('alert', 'block', 'log')) DEFAULT 'alert', 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

##### 3. 异常检测表 (anomalies)

存储所有被规则检测或AI检测出的异常日志。

```sql
CREATE TABLE anomalies (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT, 
    log_id INTEGER NOT NULL, 
    rule_id INTEGER, 
    detected_by TEXT CHECK(detected_by IN ('rules', 'AI', 'both')) NOT NULL, 
    ai_model_version TEXT, 
    score REAL, 
    anomaly_type TEXT, 
    description TEXT, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY (log_id) REFERENCES logs(log_id), 
    FOREIGN KEY (rule_id) REFERENCES rules(rule_id)
);
```

#### 字段定义

详见上述表结构中的字段定义。

#### 索引设计

```sql
CREATE INDEX idx_logs_ip ON logs(source_ip);
CREATE INDEX idx_anomalies_log_id ON anomalies(log_id);
```

### 类图/对象模型

```
┌───────────────┐      ┌───────────────────────┐
│ nlp_routes.py │      │ nlp_processor.py      │
└───────────────┘      └───────────────────────┘
       │                          │
       │                          │
       ▼                          ▼
┌────────────────┐      ┌────────────────────────┐
│ Blueprint      │      │ NL2SQLConverter        │
│ "nlp_bp"       │      ├────────────────────────┤
└────────────────┘      │ - model_name           │
       │                │ - max_tokens           │
       │                │ - temperature          │
       │                │ - table_schema         │
       │                ├────────────────────────┤
       │                │ + convert()            │
       │                │ - _preprocess_query()  │
       │                │ - _build_prompt()      │
       │                │ - _call_openrouter_api()│
       │                │ - _extract_sql()       │
       │                │ - _enhance_entity_matching()│
       │                │ - _pattern_matching_fallback()│
       │                └────────────────────────┘
       │                          │
       │                          │
       ▼                          ▼
┌────────────────┐      ┌────────────────────────┐
│ Route          │      │ 工具函数               │
│ "/nlp/query"   │      │ convert_to_sql()       │
└────────────────┘      └────────────────────────┘
```

### 算法逻辑说明

#### 1. 时间表达式识别算法

```python
def _preprocess_query(self, query_text: str) -> Tuple[str, Dict[str, Any]]:
    # 初始化时间范围字典
    time_range = {
        "type": "relative",  # 相对时间还是绝对时间范围
        "unit": "hour",      # 时间单位：hour, day, week, month
        "value": 24,         # 默认查询最近24小时
        "start": None,       # 绝对时间范围的开始时间
        "end": None          # 绝对时间范围的结束时间
    }
    
    # 处理时间表达式
    time_patterns = [
        # 最近X小时/天/周/月
        (r'(最近|过去)(\d+)(小时|天|周|月)', 
         lambda m: self._update_time_range(time_range, m.group(2), m.group(3))),
        # 今天/昨天/本周/上周/本月/上个月
        (r'(今天|昨天|本周|上周|本月|上个月)', 
         lambda m: self._handle_special_time_range(time_range, m.group(1))),
        # X小时/天前
        (r'(\d+)(小时|天)前', 
         lambda m: self._update_time_range(time_range, m.group(1), m.group(2))),
        # 从X到Y（绝对时间范围）
        (r'从(.+?)到(.+?)(的|之间)?', 
         lambda m: self._parse_absolute_time_range(time_range, m.group(1), m.group(2)))
    ]
    
    processed_text = query_text
    for pattern, handler in time_patterns:
        match = re.search(pattern, processed_text)
        if match:
            handler(match)  # 更新时间范围
            processed_text = re.sub(pattern, '在指定时间范围内', processed_text)
            
    # 其他处理逻辑...
    
    return processed_text, time_range
```

#### 2. SQL增强算法

```python
def _enhance_entity_matching(self, sql: str, processed_query: str) -> str:
    # 检查是否有UNION连接的多个查询
    if re.search(r'\bUNION\b', sql, re.IGNORECASE):
        # 使用正则表达式拆分UNION查询，同时保留UNION关键字
        parts = re.split(r'\b(UNION(?:\s+ALL)?)\b', sql, flags=re.IGNORECASE)
        
        # 分离SQL部分和UNION关键字
        sql_parts = []
        union_parts = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # 偶数索引是SQL语句部分
                sql_parts.append(part)
            else:  # 奇数索引是UNION关键字
                union_parts.append(part)
        
        # 对每个SQL部分单独进行增强
        enhanced_parts = []
        for part in sql_parts:
            if part.strip():  # 确保不是空字符串
                enhanced_part = self._enhance_single_query(part.strip(), processed_query)
                enhanced_parts.append(enhanced_part)
        
        # 重新组合增强后的查询和UNION关键字
        result = ""
        for i in range(len(enhanced_parts)):
            result += enhanced_parts[i]
            if i < len(union_parts):
                result += "\n" + union_parts[i] + "\n"
        
        return result
    else:
        # 单个查询的情况
        return self._enhance_single_query(sql, processed_query)
```

### 核心代码结构与关键实现说明

#### 文件结构

```
app/
├── services/
│   ├── nlp_processor.py   # NL2SQL转换器核心实现
├── routes/
│   ├── nlp_routes.py      # API路由和处理函数
├── models/
│   ├── db.py              # 数据库连接和表结构
├── templates/
│   ├── nlp_sql.html       # Web界面模板
```

#### 关键实现

1. **错误处理链模式**

```python
try:
    # 主要转换逻辑
    processed_query, time_range = self._preprocess_query(query_text)
    prompt = self._build_prompt(processed_query, time_range)
    result = self._call_openrouter_api(prompt)
    sql = self._extract_sql(result)
    enhanced_sql = self._enhance_entity_matching(sql, processed_query)
    
    return {
        "sql": enhanced_sql,
        "confidence": 0.9,
        "original_query": query_text
    }
except Exception as e:
    # 回退策略
    fallback_result = self._pattern_matching_fallback(processed_query, time_range)
    
    return {
        "sql": fallback_result,
        "confidence": 0.5,
        "original_query": query_text,
        "error": str(e)
    }
```

2. **API与本地模式自动切换**

```python
# 检查API密钥是否可用
api_key = os.environ.get("OPENROUTER_API_KEY", "")
if not api_key or api_key.strip() == "":
    # 如果没有API密钥，直接使用模式匹配生成简单SQL
    result = self._pattern_matching_fallback(processed_query, time_range)
    
    return {
        "sql": result,
        "confidence": 0.7,
        "original_query": query_text
    }

# 调用模型
result = self._call_openrouter_api(prompt)
```

## 七、接口文档（API 文档）

### 1. 自然语言转SQL查询接口

#### 接口路径
`/nlp/query`

#### 请求方法
`POST`

#### 请求参数
- **Content-Type**: `application/json`

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| query | String | 是 | 自然语言查询文本 |

#### 请求示例
```json
{
  "query": "最近24小时admin登录失败次数"
}
```

#### 返回值结构
```json
{
  "sql": "SELECT COUNT(*) as failure_count FROM logs WHERE timestamp >= datetime('now', '-24 hour') AND event_type = 'login' AND message LIKE '%failure%' AND user = 'admin'",
  "results": [{"failure_count": 5}],
  "original_query": "最近24小时admin登录失败次数",
  "confidence": 0.9
}
```

#### 状态码说明

| 状态码 | 说明 |
|------|------|
| 200 | 成功 |
| 400 | 请求参数错误（缺少查询参数或格式错误） |
| 500 | 服务器内部错误（SQL执行错误等） |

#### 错误响应示例
```json
{
  "error": "查询执行错误: no such column: user",
  "sql": "SELECT COUNT(*) as failure_count FROM logs WHERE timestamp >= datetime('now', '-24 hour') AND user = 'admin'",
  "original_query": "最近24小时admin登录失败次数"
}
```

### 2. Web界面接口

#### 接口路径
`/nlp/ui`

#### 请求方法
`GET`

#### 请求参数
无

#### 返回值
HTML页面，包含用户界面，用于直观地进行自然语言查询。

## 八、部署与运维

### 部署方案与步骤

#### 开发环境部署

1. 克隆项目并进入项目目录
   ```bash
   git clone <project-url>
   cd NLP-SecLogAI
   ```

2. 创建并激活虚拟环境
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量
   ```bash
   cp .env.example .env
   # 编辑.env文件，添加OPENROUTER_API_KEY
   ```

5. 运行开发服务器
   ```bash
   python run.py
   ```

#### 生产环境部署

1. 准备服务器环境
   ```bash
   # 安装Python和必要工具
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

2. 克隆项目并设置
   ```bash
   git clone <project-url>
   cd NLP-SecLogAI
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. 配置环境变量
   ```bash
   cp .env.example .env
   # 编辑并设置生产环境变量
   ```

4. 使用Gunicorn部署
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 'run:app'
   ```

5. 配置Nginx反向代理（可选）
   ```nginx
   server {
       listen 80;
       server_name your_domain.com;

       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### 运行环境配置说明

#### 环境变量配置

在`.env`文件中配置以下环境变量：

```
# API配置
OPENROUTER_API_KEY=your_api_key_here
MODEL_NAME=openrouter/auto  # 可选，默认自动选择

# 数据库配置
DATABASE_PATH=data/logs.db

# 应用配置
FLASK_ENV=development  # 开发环境
# FLASK_ENV=production  # 生产环境
DEBUG=True  # 开发时开启，生产环境关闭
```

#### 服务器配置

- **内存**: 生产环境推荐至少8GB
- **CPU**: 2核以上
- **存储**: 根据日志量决定，建议至少10GB可用空间
- **网络**: 公网访问需配置防火墙规则

### 日志、监控与告警机制

#### 日志配置

```python
import logging
from logging.handlers import RotatingFileHandler

# 配置日志
handler = RotatingFileHandler('logs/app.log', maxBytes=10485760, backupCount=10)
handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)
```

#### 监控指标

1. **API调用统计**
   - 调用次数
   - 平均响应时间
   - 错误率

2. **SQL转换质量**
   - 转换成功率
   - 回退方法使用率
   - 用户反馈评分

3. **系统资源**
   - CPU使用率
   - 内存使用率
   - 磁盘空间

#### 告警机制

1. **错误率告警**
   - 当转换错误率超过10%时触发告警

2. **响应时间告警**
   - 当平均响应时间超过5秒时触发告警

3. **资源告警**
   - 当CPU使用率超过80%时触发告警
   - 当可用磁盘空间低于10%时触发告警

### 备份与恢复策略

#### 数据库备份

1. **定时备份**
   ```bash
   # 每日备份脚本
   #!/bin/bash
   BACKUP_DIR="/backups"
   DATE=$(date +%Y%m%d)
   sqlite3 /path/to/data/logs.db ".backup $BACKUP_DIR/logs_$DATE.db"
   ```

2. **备份轮换**
   - 保留最近30天的每日备份
   - 每月第一天的备份保留一年

#### 恢复流程

1. **从备份恢复**
   ```bash
   # 恢复数据库
   sqlite3 /path/to/data/logs.db ".restore /backups/logs_20230401.db"
   ```

2. **系统配置恢复**
   - 保存`.env`文件和配置的备份
   - 记录系统配置变更

## 九、测试说明

### 测试策略与测试类型

#### 单元测试

测试各个组件的独立功能：

1. **查询预处理测试**
   - 时间表达式识别
   - 实体识别
   - 查询意图分析

2. **SQL转换测试**
   - API调用（使用mock）
   - SQL提取
   - 回退方法

3. **实体增强测试**
   - 各种实体类型的增强

#### 集成测试

测试组件间的交互：

1. **端到端流程**
   - 从用户输入到SQL生成
   - 从SQL生成到结果返回

2. **错误处理**
   - API调用失败
   - SQL执行错误

#### 性能测试

1. **响应时间**
   - 普通查询
   - 复杂查询

2. **并发能力**
   - 多用户同时请求

### 测试环境配置

#### 单元测试环境
