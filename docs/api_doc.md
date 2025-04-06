# 五、接口文档

## 5.1 日志接口

日志接口用于上传、查询和管理安全日志数据。

### 5.1.1 日志上传

#### 请求方法
`POST /logs/upload`

#### 功能说明
上传并解析日志文件

#### 请求参数
- **Content-Type**: `multipart/form-data`

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 日志文件 |
| log_type | String | 否 | 日志类型，可选值：ssh、web、firewall、mysql、hdfs、linux。如不提供，系统将尝试自动检测 |

#### 响应结果
```json
{
  "success": true,
  "processed": 150,
  "saved": 150
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| success | Boolean | 是否处理成功 |
| processed | Integer | 处理的日志条目数量 |
| saved | Integer | 成功保存到数据库的条目数量 |

#### 错误响应
```json
{
  "error": "未提供文件"
}
```

### 5.1.2 日志列表获取

#### 请求方法
`GET /logs`

#### 功能说明
分页获取日志列表

#### 请求参数
- **Content-Type**: `application/json`

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | Integer | 否 | 页码，默认1 |
| limit | Integer | 否 | 每页记录数，默认50 |
| log_type | String | 否 | 过滤特定类型的日志 |
| source_ip | String | 否 | 按源IP过滤 |
| start_time | String | 否 | 开始时间，格式：YYYY-MM-DD HH:MM:SS |
| end_time | String | 否 | 结束时间，格式：YYYY-MM-DD HH:MM:SS |
| event_type | String | 否 | 事件类型过滤 |

#### 响应结果
```json
{
  "logs": [
    {
      "log_id": 1,
      "timestamp": "2023-04-10 13:45:27",
      "source_ip": "192.168.1.100",
      "event_type": "login",
      "message": "Accepted password for user1 from 192.168.1.100 port 22 ssh2",
      "detected_by": "none",
      "created_at": "2023-04-10 13:46:30"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 50,
    "total": 150
  }
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| logs | Array | 日志记录数组 |
| meta.page | Integer | 当前页码 |
| meta.limit | Integer | 每页记录数 |
| meta.total | Integer | 总记录数 |

### 5.1.3 日志详情获取

#### 请求方法
`GET /logs/{log_id}`

#### 功能说明
获取特定日志条目详情

#### 请求参数
- **路径参数**：

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| log_id | Integer | 是 | 日志ID |

#### 响应结果
```json
{
  "log": {
    "log_id": 1,
    "timestamp": "2023-04-10 13:45:27",
    "source_ip": "192.168.1.100",
    "event_type": "login",
    "message": "Accepted password for user1 from 192.168.1.100 port 22 ssh2",
    "detected_by": "none",
    "created_at": "2023-04-10 13:46:30",
    "parsed_data": {
      "user": "user1",
      "port": 22,
      "protocol": "ssh2"
    },
    "related_anomalies": []
  }
}
```

#### 错误响应
```json
{
  "error": "日志不存在"
}
```

## 5.2 NLP分析接口

NLP分析接口用于对日志进行自然语言处理分析。

### 5.2.1 日志文本分析

#### 请求方法
`POST /nlp/analyze`

#### 功能说明
对提供的文本进行NLP分析

#### 请求参数
- **Content-Type**: `application/json`

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| text | String | 是 | 要分析的日志文本 |
| analysis_type | String | 否 | 分析类型，可选值：full、entities、keywords。默认为full |

#### 请求示例
```json
{
  "text": "Failed password for invalid user admin from 10.0.0.1 port 22 ssh2",
  "analysis_type": "full"
}
```

#### 响应结果
```json
{
  "entities": [
    {
      "text": "Failed password",
      "type": "auth_event",
      "start": 0,
      "end": 15
    },
    {
      "text": "admin",
      "type": "user",
      "start": 27,
      "end": 32
    },
    {
      "text": "10.0.0.1",
      "type": "ip_address",
      "start": 38,
      "end": 46
    },
    {
      "text": "22",
      "type": "port",
      "start": 53,
      "end": 55
    }
  ],
  "keywords": ["failed", "password", "invalid", "admin"],
  "sentiment": {
    "score": -0.3,
    "label": "negative"
  }
}
```

### 5.2.2 日志批量分析

#### 请求方法
`POST /nlp/batch-analyze`

#### 功能说明
批量分析多条日志文本

#### 请求参数
- **Content-Type**: `application/json`

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| texts | Array(String) | 是 | 要分析的日志文本数组 |
| analysis_type | String | 否 | 分析类型，可选值同上 |

#### 响应结果
```json
{
  "results": [
    {
      "text": "原始文本1",
      "entities": [...],
      "keywords": [...]
    },
    {
      "text": "原始文本2",
      "entities": [...],
      "keywords": [...]
    }
  ]
}
```

## 5.3 异常检测接口

异常检测接口用于执行各种安全日志的异常检测和查询异常结果。

### 5.3.1 执行异常检测

#### 请求方法
`GET /anomalies/detect/{log_type}`

#### 功能说明
对指定类型的日志执行异常检测规则

#### 请求参数
- **路径参数**：

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| log_type | String | 是 | 日志类型，可选值：ssh、web、firewall、mysql、hdfs、linux、all |

#### 响应结果
```json
{
  "ssh_anomalies": [
    {
      "source_ip": "192.168.1.100",
      "user": "root",
      "type": "brute_force",
      "reason": "检测到SSH暴力破解尝试: 6次失败登录"
    }
  ],
  "count": 1
}
```

### 5.3.2 获取异常列表

#### 请求方法
`GET /anomalies`

#### 功能说明
分页获取异常列表

#### 请求参数
- **查询参数**：

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | Integer | 否 | 页码，默认1 |
| limit | Integer | 否 | 每页记录数，默认20 |
| anomaly_type | String | 否 | 异常类型过滤 |
| detected_by | String | 否 | 检测方式过滤，可选值：rules、AI、both |
| start_time | String | 否 | 开始时间，格式：YYYY-MM-DD HH:MM:SS |
| end_time | String | 否 | 结束时间，格式：YYYY-MM-DD HH:MM:SS |

#### 响应结果
```json
{
  "anomalies": [
    {
      "anomaly_id": 1,
      "log_id": 42,
      "rule_id": 3,
      "detected_by": "rules",
      "anomaly_type": "brute_force",
      "score": 1.0,
      "description": "检测到SSH暴力破解尝试",
      "created_at": "2023-04-10 14:22:35"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 32
  }
}
```

### 5.3.3 获取异常详情

#### 请求方法
`GET /anomalies/{anomaly_id}`

#### 功能说明
获取特定异常的详细信息

#### 请求参数
- **路径参数**：

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| anomaly_id | Integer | 是 | 异常ID |

#### 响应结果
```json
{
  "anomaly": {
    "anomaly_id": 1,
    "log_id": 42,
    "rule_id": 3,
    "detected_by": "rules",
    "anomaly_type": "brute_force",
    "score": 1.0,
    "description": "检测到SSH暴力破解尝试",
    "created_at": "2023-04-10 14:22:35",
    "log": {
      "log_id": 42,
      "message": "Failed password for root from 192.168.1.100 port 22 ssh2",
      "timestamp": "2023-04-10 14:22:30"
    },
    "rule": {
      "rule_id": 3,
      "name": "SSH暴力破解检测",
      "description": "检测短时间内多次SSH登录失败尝试"
    },
    "related_anomalies": [
      {
        "anomaly_id": 2,
        "anomaly_type": "brute_force",
        "created_at": "2023-04-10 14:22:40"
      }
    ]
  }
}
```

## 5.4 前端API接口

前端API接口为Web界面提供数据支持，包括仪表盘数据、统计信息和配置管理等。

### 5.4.1 获取仪表盘数据

#### 请求方法
`GET /api/dashboard`

#### 功能说明
获取仪表盘概览数据

#### 请求参数
- **查询参数**：

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| time_range | String | 否 | 时间范围，可选值：today、week、month，默认today |

#### 响应结果
```json
{
  "anomalies_count": {
    "total": 156,
    "high": 23,
    "medium": 85,
    "low": 48
  },
  "logs_count": {
    "total": 12503,
    "ssh": 3200,
    "web": 5600,
    "firewall": 2100,
    "others": 1603
  },
  "anomalies_by_type": [
    {
      "type": "brute_force",
      "count": 42
    },
    {
      "type": "sql_injection",
      "count": 23
    },
    {
      "type": "xss",
      "count": 15
    }
  ],
  "top_attacked_ips": [
    {
      "ip": "192.168.1.10",
      "count": 56
    },
    {
      "ip": "192.168.1.15",
      "count": 34
    }
  ],
  "recent_anomalies": [
    {
      "anomaly_id": 1023,
      "anomaly_type": "sql_injection",
      "source_ip": "192.168.1.100",
      "timestamp": "2023-04-10 14:35:27"
    }
  ]
}
```

### 5.4.2 获取规则列表

#### 请求方法
`GET /api/rules`

#### 功能说明
获取异常检测规则列表

#### 请求参数
- **查询参数**：

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| page | Integer | 否 | 页码，默认1 |
| limit | Integer | 否 | 每页记录数，默认20 |

#### 响应结果
```json
{
  "rules": [
    {
      "rule_id": 1,
      "name": "SSH暴力破解检测",
      "description": "检测短时间内多次SSH登录失败尝试",
      "action": "alert",
      "created_at": "2023-03-15 10:22:30"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 45
  }
}
```

### 5.4.3 添加/修改规则

#### 请求方法
`POST /api/rules` (新增)  
`PUT /api/rules/{rule_id}` (修改)

#### 功能说明
添加或修改异常检测规则

#### 请求参数
- **Content-Type**: `application/json`

| 参数名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | String | 是 | 规则名称 |
| description | String | 是 | 规则描述 |
| sql_query | String | 是 | 规则SQL查询 |
| action | String | 是 | 规则动作，可选值：alert、block、log |

#### 请求示例
```json
{
  "name": "SSH暴力破解检测",
  "description": "检测短时间内多次SSH登录失败尝试",
  "sql_query": "SELECT source_ip, user, COUNT(*) as attempts FROM ssh_logs WHERE event_type = 'failed_login' AND timestamp >= datetime('now', '-300 seconds') GROUP BY source_ip, user HAVING attempts >= 5",
  "action": "alert"
}
```

#### 响应结果
```json
{
  "success": true,
  "rule_id": 46
}
```

### 5.4.4 系统状态接口

#### 请求方法
`GET /api/system/status`

#### 功能说明
获取系统运行状态信息

#### 响应结果
```json
{
  "status": "running",
  "uptime": "3 days, 5 hours",
  "version": "0.2.0-alpha",
  "db_size": "156 MB",
  "logs_count": 12503,
  "anomalies_count": 156,
  "rules_count": 45,
  "last_detection_run": "2023-04-10 14:45:30",
  "system_resources": {
    "cpu_usage": "23%",
    "memory_usage": "512MB / 2GB",
    "disk_usage": "1.2GB / 20GB"
  }
}
``` 