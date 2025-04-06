# 九、附录

## 9.1 术语表

| 术语 | 定义 |
|-----|-----|
| NLP | 自然语言处理（Natural Language Processing），计算机科学和人工智能的一个领域，使计算机能够理解、解释和生成人类语言 |
| 异常检测 | 识别与预期模式显著不同的项目、事件或观察值的过程 |
| 日志解析 | 将非结构化或半结构化的日志文本转换为结构化数据的过程 |
| 实体识别 | 从文本中识别和提取特定类型的实体（如用户名、IP地址等）的NLP技术 |
| 规则引擎 | 基于一组预定义规则对数据进行分析和决策的系统组件 |
| 误报 | 系统错误地将正常行为标识为异常或威胁的情况 |
| 漏报 | 系统未能检测到实际存在的异常或威胁的情况 |
| SSH | 安全外壳协议（Secure Shell），一种加密网络协议，用于在不安全网络上安全地操作网络服务 |
| SQL注入 | 一种代码注入技术，攻击者通过在用户输入中插入SQL语句来操纵后端数据库 |
| XSS | 跨站脚本攻击（Cross-Site Scripting），攻击者将恶意脚本注入到受信任的网站中 |
| 暴力破解 | 攻击者通过尝试大量可能的密码或密钥来获取访问权限的方法 |
| CSRF | 跨站请求伪造（Cross-Site Request Forgery），攻击者诱导用户执行非本意的操作 |
| 威胁情报 | 关于现有或潜在威胁的已收集、处理和分析的证据和知识 |
| 零日漏洞 | 软件、硬件或固件中尚未被官方修复的漏洞 |
| ETL | 提取、转换、加载（Extract, Transform, Load），数据仓库数据处理的一个过程 |
| API | 应用程序接口（Application Programming Interface），允许不同软件组件相互交互的接口 |
| 沙箱 | 一种安全机制，提供隔离环境运行未经验证或不受信任的程序 |
| RBAC | 基于角色的访问控制（Role-Based Access Control），根据用户在组织中的角色限制系统访问 |
| 防火墙日志 | 记录防火墙允许或拒绝的网络流量的日志 |
| SIEM | 安全信息和事件管理（Security Information and Event Management），提供实时分析安全警报的系统 |

## 9.2 日志格式参考

### 9.2.1 SSH日志格式

SSH日志通常由系统的sshd守护进程生成，记录了SSH连接和认证尝试的信息。

**标准格式**：
```
<时间戳> <主机名> sshd[<进程ID>]: <消息内容>
```

**示例**：
```
Apr 10 13:45:27 server sshd[12345]: Accepted password for user1 from 192.168.1.100 port 22 ssh2
Apr 10 13:46:13 server sshd[12346]: Failed password for invalid user admin from 10.0.0.1 port 22 ssh2
Apr 10 13:47:02 server sshd[12347]: Connection closed by 192.168.1.101 port 49812
```

**常见事件类型**：
- 登录成功：`Accepted password for <用户> from <IP> port <端口> ssh2`
- 登录失败：`Failed password for [invalid user] <用户> from <IP> port <端口> ssh2`
- 连接关闭：`Connection closed by <IP> port <端口>`
- 会话开始：`pam_unix(sshd:session): session opened for user <用户> by (uid=0)`
- 会话结束：`pam_unix(sshd:session): session closed for user <用户>`

### 9.2.2 Web服务器日志格式

#### Apache访问日志（Common Log Format）

**格式**：
```
<客户端IP> <身份标识> <用户名> [<时间>] "<请求方法> <请求URL> <协议>" <状态码> <响应大小>
```

**示例**：
```
192.168.1.100 - - [10/Apr/2023:14:15:16 +0000] "GET /index.html HTTP/1.1" 200 1234
10.0.0.1 - - [10/Apr/2023:14:16:22 +0000] "POST /login.php HTTP/1.1" 302 0
192.168.1.101 - - [10/Apr/2023:14:17:35 +0000] "GET /admin HTTP/1.1" 403 567
```

#### Nginx访问日志

**格式**：
```
<客户端IP> - <远程用户> [<时间>] "<请求方法> <请求URL> <协议>" <状态码> <响应大小> "<引用页>" "<用户代理>"
```

**示例**：
```
192.168.1.100 - - [10/Apr/2023:14:15:16 +0000] "GET /index.html HTTP/1.1" 200 1234 "https://example.com" "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
```

### 9.2.3 防火墙日志格式

#### iptables日志

**格式**：
```
<时间戳> <主机名> kernel: [<时间>] IPTABLES: <链> <详细信息> SRC=<源IP> DST=<目标IP> LEN=<长度> TOS=<服务类型> PREC=<优先级> TTL=<生命周期> ID=<ID> PROTO=<协议> SPT=<源端口> DPT=<目标端口>
```

**示例**：
```
Apr 10 14:20:45 firewall kernel: [14323.456789] IPTABLES: INPUT DROP IN=eth0 OUT= MAC=00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd SRC=10.0.0.2 DST=192.168.1.1 LEN=60 TOS=0x00 PREC=0x00 TTL=64 ID=12345 PROTO=TCP SPT=45678 DPT=22 WINDOW=29200 RES=0x00 SYN URGP=0
```

#### pfSense日志

**格式**：
```
<时间戳> <规则编号> <动作> <接口> <协议> <源IP>:<源端口> <目标IP>:<目标端口>
```

**示例**：
```
Apr 10 14:25:12 filterlog: 5,16777216,1000,em0,match,block,in,4,0x0,,64,12345,0,DF,6,tcp,60,10.0.0.2,192.168.1.1,45678,22,0,S,12345,0,0,0,60,
```

## 9.3 异常检测规则示例

### 9.3.1 SSH暴力破解检测

**规则描述**：检测短时间内多次SSH登录失败尝试

**SQL查询**：
```sql
SELECT source_ip, user, COUNT(*) as attempts 
FROM ssh_logs 
WHERE event_type = 'failed_login' 
AND timestamp >= datetime('now', '-300 seconds') 
GROUP BY source_ip, user 
HAVING attempts >= 5
```

### 9.3.2 可疑用户检测

**规则描述**：检测登录尝试涉及敏感用户名的情况

**SQL查询**：
```sql
SELECT * FROM ssh_logs 
WHERE user IN ('root', 'admin', 'administrator', 'oracle', 'postgres')
AND source_ip NOT IN (SELECT ip FROM whitelist)
```

### 9.3.3 Web扫描检测

**规则描述**：检测短时间内过多的404错误

**SQL查询**：
```sql
SELECT source_ip, COUNT(*) as attempts 
FROM web_logs 
WHERE status_code = '404' 
AND timestamp >= datetime('now', '-60 seconds') 
GROUP BY source_ip 
HAVING attempts >= 20
```

### 9.3.4 SQL注入尝试检测

**规则描述**：检测URL中可能包含SQL注入语法的请求

**正则表达式**：
```
/((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))/i
```

或

```sql
SELECT * FROM web_logs 
WHERE request_url LIKE '%27%' 
OR request_url LIKE '%\'%' 
OR request_url LIKE '%--+%' 
OR request_url LIKE '%union%select%'
```

### 9.3.5 异常登录时间检测

**规则描述**：检测非工作时间的登录活动

**SQL查询**：
```sql
SELECT * FROM ssh_logs 
WHERE event_type = 'login' 
AND (CAST(strftime('%H', timestamp) AS INTEGER) < 8 
OR CAST(strftime('%H', timestamp) AS INTEGER) > 18) 
AND user IN (SELECT username FROM employees WHERE role != 'it_admin')
```

## 9.4 性能基准测试结果

下表展示了系统在不同负载条件下的性能测试结果：

### 9.4.1 单日志解析性能

| 日志类型 | 每秒解析数量 | CPU使用率 | 内存使用 |
|---------|-----------|---------|---------|
| SSH日志 | 5,000 | 15% | 120MB |
| Web访问日志 | 8,000 | 25% | 180MB |
| 防火墙日志 | 4,000 | 20% | 150MB |

### 9.4.2 批量日志处理性能

| 日志文件大小 | 处理时间 | CPU使用率 | 内存使用 |
|------------|---------|---------|---------|
| 10MB (约50,000条) | 12秒 | 45% | 250MB |
| 100MB (约500,000条) | 110秒 | 60% | 500MB |
| 1GB (约5,000,000条) | 18分钟 | 75% | 1.2GB |

### 9.4.3 异常检测性能

| 日志数量 | 规则数量 | 检测时间 | CPU使用率 | 内存使用 |
|---------|---------|---------|---------|---------|
| 10,000 | 10 | 2秒 | 30% | 200MB |
| 100,000 | 20 | 12秒 | 50% | 400MB |
| 1,000,000 | 30 | 95秒 | 70% | 800MB |

### 9.4.4 并发用户测试

| 并发用户数 | 响应时间 (平均) | 响应时间 (95%分位) | 错误率 |
|----------|--------------|-----------------|-------|
| 10 | 0.3秒 | 0.5秒 | 0% |
| 50 | 0.8秒 | 1.5秒 | 0% |
| 100 | 1.5秒 | 2.8秒 | 0.5% |
| 200 | 3.0秒 | 5.5秒 | 2% |

## 9.5 数据库表结构

### 9.5.1 核心表结构

#### logs表

```sql
CREATE TABLE logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    source_ip VARCHAR(45) NOT NULL,
    log_type VARCHAR(20) NOT NULL,
    event_type VARCHAR(50),
    message TEXT NOT NULL,
    parsed_data JSON,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_logs_timestamp ON logs(timestamp);
CREATE INDEX idx_logs_source_ip ON logs(source_ip);
CREATE INDEX idx_logs_log_type ON logs(log_type);
CREATE INDEX idx_logs_event_type ON logs(event_type);
```

#### ssh_logs表

```sql
CREATE TABLE ssh_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    host VARCHAR(100) NOT NULL,
    process_id INTEGER,
    source_ip VARCHAR(45) NOT NULL,
    port INTEGER,
    user VARCHAR(50),
    event_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    auth_method VARCHAR(20),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (log_id) REFERENCES logs(log_id) ON DELETE CASCADE
);

CREATE INDEX idx_ssh_logs_user ON ssh_logs(user);
CREATE INDEX idx_ssh_logs_source_ip_event_type ON ssh_logs(source_ip, event_type);
```

#### web_logs表

```sql
CREATE TABLE web_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    source_ip VARCHAR(45) NOT NULL,
    method VARCHAR(10) NOT NULL,
    path TEXT NOT NULL,
    protocol VARCHAR(20) NOT NULL,
    status_code INTEGER NOT NULL,
    size INTEGER,
    referer TEXT,
    user_agent TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (log_id) REFERENCES logs(log_id) ON DELETE CASCADE
);

CREATE INDEX idx_web_logs_source_ip ON web_logs(source_ip);
CREATE INDEX idx_web_logs_status_code ON web_logs(status_code);
CREATE INDEX idx_web_logs_path ON web_logs(path);
```

#### anomalies表

```sql
CREATE TABLE anomalies (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_id INTEGER,
    rule_id INTEGER,
    detection_method VARCHAR(20) NOT NULL,
    anomaly_type VARCHAR(50) NOT NULL,
    source_ip VARCHAR(45),
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    score FLOAT,
    description TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (log_id) REFERENCES logs(log_id) ON DELETE SET NULL,
    FOREIGN KEY (rule_id) REFERENCES rules(rule_id) ON DELETE SET NULL
);

CREATE INDEX idx_anomalies_source_ip ON anomalies(source_ip);
CREATE INDEX idx_anomalies_anomaly_type ON anomalies(anomaly_type);
CREATE INDEX idx_anomalies_detection_method ON anomalies(detection_method);
CREATE INDEX idx_anomalies_severity ON anomalies(severity);
```

#### rules表

```sql
CREATE TABLE rules (
    rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    rule_type VARCHAR(20) NOT NULL,
    log_type VARCHAR(20) NOT NULL,
    condition_data JSON NOT NULL,
    severity VARCHAR(20) NOT NULL DEFAULT 'medium',
    enabled BOOLEAN NOT NULL DEFAULT 1,
    action VARCHAR(20) NOT NULL DEFAULT 'alert',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_rules_log_type ON rules(log_type);
CREATE INDEX idx_rules_enabled ON rules(enabled);
```

### 9.5.2 用户和权限表

#### users表

```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role_id INTEGER NOT NULL,
    last_login DATETIME,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (role_id) REFERENCES roles(role_id)
);
```

#### roles表

```sql
CREATE TABLE roles (
    role_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name VARCHAR(50) NOT NULL UNIQUE,
    description TEXT,
    permissions JSON NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
```

## 9.6 API参考

### 9.6.1 认证

所有API请求（除了登录API）都需要在HTTP头部包含认证令牌：

```
Authorization: Bearer <your_api_token>
```

#### 获取API令牌

**请求**：
```
POST /api/auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**响应**：
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_at": "2023-05-10T14:30:00Z",
  "user": {
    "user_id": 1,
    "username": "your_username",
    "role": "admin"
  }
}
```

### 9.6.2 错误处理

API错误返回标准HTTP状态码和JSON格式的错误详情：

```json
{
  "error": "错误类型",
  "message": "错误详细信息",
  "status_code": 400
}
```

常见状态码：
- 200: 请求成功
- 400: 错误请求
- 401: 未认证
- 403: 权限不足
- 404: 资源不存在
- 500: 服务器内部错误

### 9.6.3 分页

所有返回列表的API都支持分页，使用以下查询参数：

- `page`: 页码，默认为1
- `limit`: 每页记录数，默认为20，最大100

分页响应包含元数据：

```json
{
  "data": [...],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 42
  }
}
```

## 9.7 第三方库和依赖

| 库名 | 版本 | 用途 | 许可证 |
|-----|-----|-----|------|
| Flask | 2.0.1 | Web框架 | BSD-3-Clause |
| SQLAlchemy | 1.4.20 | ORM和数据库交互 | MIT |
| spaCy | 3.2.0 | NLP处理 | MIT |
| pandas | 1.3.3 | 数据处理和分析 | BSD-3-Clause |
| scikit-learn | 1.0.0 | 机器学习 | BSD-3-Clause |
| PyTorch | 1.10.0 | 深度学习 | BSD-3-Clause |
| React | 17.0.2 | 前端UI库 | MIT |
| Redux | 4.1.1 | 前端状态管理 | MIT |
| Bootstrap | 5.1.0 | CSS框架 | MIT |
| Chart.js | 3.5.1 | 图表生成 | MIT |
| Celery | 5.1.2 | 分布式任务队列 | BSD-3-Clause |
| Redis | 3.5.3 | 缓存和消息代理 | MIT |
| pytest | 6.2.5 | 测试框架 | MIT |
| Alembic | 1.7.1 | 数据库迁移 | MIT |
| Flask-JWT-Extended | 4.3.1 | JWT认证 | MIT |
| Gunicorn | 20.1.0 | WSGI HTTP服务器 | MIT |
| Nginx | 1.20.0 | Web服务器 | 2-clause BSD |
| PostgreSQL | 13.4 | 数据库 | PostgreSQL |
| Docker | 20.10.8 | 容器化 | Apache-2.0 |

## 9.8 相关资源和链接

### 9.8.1 官方资源

- 项目GitHub仓库: [https://github.com/yourusername/NLP-SecLogAI](https://github.com/yourusername/NLP-SecLogAI)
- 项目文档: [https://docs.seclogai.com](https://docs.seclogai.com)
- API文档: [https://docs.seclogai.com/api](https://docs.seclogai.com/api)
- 问题跟踪: [https://github.com/yourusername/NLP-SecLogAI/issues](https://github.com/yourusername/NLP-SecLogAI/issues)

### 9.8.2 安全日志分析资源

- OWASP日志指南: [https://owasp.org/www-community/attacks/](https://owasp.org/www-community/attacks/)
- SANS日志管理: [https://www.sans.org/reading-room/whitepapers/logging/](https://www.sans.org/reading-room/whitepapers/logging/)
- NIST安全日志管理指南: [https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-92.pdf](https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-92.pdf)

### 9.8.3 NLP和机器学习资源

- spaCy文档: [https://spacy.io/api](https://spacy.io/api)
- scikit-learn文档: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)
- 安全领域的NLP应用论文: [https://arxiv.org/abs/2103.00635](https://arxiv.org/abs/2103.00635) 