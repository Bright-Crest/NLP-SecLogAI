# 四、详细设计

## 4.1 数据库设计

### 数据库概述

NLP-SecLogAI系统采用混合存储策略，主要使用SQLite作为关系型数据库存储结构化数据。数据库设计支持扩展至PostgreSQL以适应生产环境需求。数据库主要存储用户信息、日志数据、分析结果和系统配置等信息。

### 数据模型关系图

```
logs (1) --- (N) anomalies  ✅ 一个日志可能触发多个异常
rules (1) --- (N) anomalies  ✅ 一个规则可能触发多个异常
```

### 核心表结构设计

#### 日志表 (logs)

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

| 字段 | 类型 | 说明 |
|------|------|------|
| log_id | INTEGER | 日志唯一ID |
| timestamp | DATETIME | 日志时间 |
| source_ip | TEXT | 来源IP |
| event_type | TEXT | 事件类型，如login、file_access |
| message | TEXT | 日志内容 |
| detected_by | TEXT | 规则检测/AI检测标记 |
| created_at | TIMESTAMP | 记录创建时间 |

#### 规则表 (rules)

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

| 字段 | 类型 | 说明 |
|------|------|------|
| rule_id | INTEGER | 规则唯一ID |
| name | TEXT | 规则名称 |
| description | TEXT | 规则描述 |
| sql_query | TEXT | 规则SQL查询 |
| action | TEXT | alert（警告）、block（拦截）、log（仅记录） |
| created_at | TIMESTAMP | 规则创建时间 |

#### 异常检测表 (anomalies)

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

| 字段 | 类型 | 说明 |
|------|------|------|
| anomaly_id | INTEGER | 异常唯一ID |
| log_id | INTEGER | 关联的日志ID |
| rule_id | INTEGER | 关联的规则ID（若由规则检测） |
| detected_by | TEXT | rules / AI / both |
| ai_model_version | TEXT | 触发该异常的AI版本 |
| score | REAL | AI置信度 |
| anomaly_type | TEXT | 异常类别 |
| description | TEXT | 异常详细描述 |
| created_at | TIMESTAMP | 记录创建时间 |

#### 特定日志类型表

系统还包含针对特定日志类型的专用表，例如：

##### SSH日志表 (ssh_logs)

```sql
CREATE TABLE ssh_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    process_id INTEGER,
    event_type TEXT NOT NULL,
    user TEXT,
    source_ip TEXT,
    port INTEGER,
    country_code TEXT,
    forwarded_ports TEXT
);
```

##### Web日志表 (web_logs)

```sql
CREATE TABLE web_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME NOT NULL,
    source_ip TEXT NOT NULL,
    method TEXT,
    path TEXT,
    status_code INTEGER,
    user_agent TEXT
);
```

### 索引设计

为提高查询性能，系统在关键字段上设置索引：

```sql
CREATE INDEX idx_logs_ip ON logs(source_ip);
CREATE INDEX idx_anomalies_log_id ON anomalies(log_id);
CREATE INDEX idx_ssh_logs_source_ip ON ssh_logs(source_ip);
CREATE INDEX idx_web_logs_source_ip ON web_logs(source_ip);
```

### 数据库迁移策略

系统支持使用Flask-Migrate进行数据库结构变更管理：

1. 初始化数据库：
   ```bash
   flask db init
   flask db migrate -m "Initial database structure"
   flask db upgrade
   ```

2. 结构变更：
   ```bash
   flask db migrate -m "Add new field"
   flask db upgrade
   ```

## 4.2 类图/对象模型

### 核心模块类关系图

```
┌──────────────────┐     ┌──────────────────┐
│   LogParser      │<────│  ParserFactory   │
└───────┬──────────┘     └──────────────────┘
        │
        ▼
┌───────────────────┐    ┌──────────────────┐
│ LogParserStrategy │<───┤ SSHLogParser     │
└───────────────────┘    ├──────────────────┤
        ▲                │ WebLogParser     │
        │                ├──────────────────┤
        │                │ FirewallParser   │
        │                └──────────────────┘
        │
┌───────┴──────────┐     ┌──────────────────┐
│  NLPProcessor    │────>│ EntityExtractor  │
└───────┬──────────┘     └──────────────────┘
        │
        ▼
┌───────────────────┐    ┌──────────────────┐
│  AnomalyDetector  │<───┤ RuleEngine       │
└───────────────────┘    ├──────────────────┤
        ▲                │ AIDetector       │
        │                └──────────────────┘
        │
┌───────┴──────────┐
│  AlertManager    │
└──────────────────┘
```

### 关键类详细设计

#### LogParser类

负责解析和处理日志数据的基础类。

```python
class LogParser:
    """日志解析器基类，处理日志收集和预处理"""
    
    def __init__(self, config=None):
        """初始化解析器"""
        self.config = config or {}
        self.parsers = {}
        self._init_parsers()
    
    def _init_parsers(self):
        """初始化所有解析器策略"""
        # 实现初始化不同类型的解析器
        
    def parse(self, log_data, log_type=None):
        """
        解析日志数据
        
        Args:
            log_data: 原始日志数据
            log_type: 日志类型，如果为None则自动检测
            
        Returns:
            解析后的结构化日志数据
        """
        # 实现解析逻辑
        
    def detect_log_type(self, log_data):
        """
        检测日志类型
        
        Args:
            log_data: 原始日志数据
            
        Returns:
            检测到的日志类型
        """
        # 实现日志类型检测逻辑
```

#### LogParserStrategy接口

定义日志解析器策略接口。

```python
class LogParserStrategy:
    """日志解析策略接口"""
    
    def parse(self, log_data):
        """
        解析日志数据
        
        Args:
            log_data: 原始日志数据
            
        Returns:
            解析后的结构化日志数据，失败返回None
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def validate(self, log_data):
        """
        验证日志数据是否符合此解析器处理的格式
        
        Args:
            log_data: 原始日志数据
            
        Returns:
            布尔值，表示是否可以处理
        """
        raise NotImplementedError("子类必须实现此方法")
```

#### NLPProcessor类

NLP处理引擎核心类。

```python
class NLPProcessor:
    """NLP处理引擎，负责文本分析和实体提取"""
    
    def __init__(self, model_path=None):
        """
        初始化NLP处理器
        
        Args:
            model_path: NLP模型路径
        """
        self.model_path = model_path
        self.nlp = self._load_model()
        self.entity_extractor = EntityExtractor(self.nlp)
    
    def _load_model(self):
        """加载NLP模型"""
        # 实现模型加载逻辑
        
    def process(self, text):
        """
        处理文本数据
        
        Args:
            text: 输入文本
            
        Returns:
            处理结果字典
        """
        # 实现NLP处理逻辑
        
    def extract_entities(self, text):
        """
        提取命名实体
        
        Args:
            text: 输入文本
            
        Returns:
            实体列表
        """
        return self.entity_extractor.extract(text)
```

#### AnomalyDetector类

异常检测器基础类。

```python
class AnomalyDetector:
    """异常检测器基类，协调规则检测和AI检测"""
    
    def __init__(self, db_conn=None):
        """
        初始化异常检测器
        
        Args:
            db_conn: 数据库连接
        """
        self.db_conn = db_conn
        self.rule_engine = RuleEngine(db_conn)
        self.ai_detector = AIDetector()
    
    def detect(self, log_data, nlp_results=None):
        """
        检测日志异常
        
        Args:
            log_data: 日志数据
            nlp_results: NLP分析结果(可选)
            
        Returns:
            异常检测结果列表
        """
        # 规则检测
        rule_results = self.rule_engine.detect(log_data)
        
        # AI检测
        ai_results = self.ai_detector.detect(log_data, nlp_results)
        
        # 合并结果
        return self._merge_results(rule_results, ai_results)
    
    def _merge_results(self, rule_results, ai_results):
        """合并规则检测和AI检测结果"""
        # 实现结果合并逻辑
```

## 4.3 算法逻辑说明

### 日志解析算法

#### 日志类型自动检测

系统使用特征匹配和正则表达式的组合来自动识别日志类型：

1. 首先尝试使用每种已知日志类型的解析器对日志进行验证
2. 每个解析器使用特定模式匹配尝试识别日志格式
3. 返回匹配度最高的解析器类型

```python
def detect_log_type(log_data):
    """检测日志类型"""
    max_score = 0
    detected_type = None
    
    for parser_type, parser in parsers.items():
        score = parser.validate(log_data)
        if score > max_score:
            max_score = score
            detected_type = parser_type
    
    return detected_type if max_score > THRESHOLD else None
```

#### 字段提取逻辑

对于每种日志类型，系统使用特定的正则表达式模式来提取关键字段：

```python
def extract_ssh_fields(log_line):
    """从SSH日志中提取字段"""
    # 示例SSH日志模式
    pattern = r'(\w+\s+\d+\s+\d+:\d+:\d+)\s+\S+\s+sshd\[(\d+)\]:\s+(.*)'
    match = re.match(pattern, log_line)
    
    if not match:
        return None
        
    timestamp, pid, message = match.groups()
    
    # 进一步解析message部分提取更多字段
    user = extract_user(message)
    ip = extract_ip(message)
    
    return {
        'timestamp': parse_timestamp(timestamp),
        'process_id': int(pid),
        'message': message,
        'user': user,
        'source_ip': ip
    }
```

### NLP处理算法

#### 实体识别

系统使用基于规则和机器学习的混合方法来识别日志中的实体：

```python
def extract_entities(text):
    """提取日志中的实体"""
    entities = []
    
    # 规则识别
    entities.extend(extract_ips(text))
    entities.extend(extract_paths(text))
    entities.extend(extract_users(text))
    
    # 机器学习识别
    doc = nlp_model(text)
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'type': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'method': 'ml'
        })
    
    # 合并结果，去除重复
    return merge_entities(entities)
```

#### 文本特征提取

系统使用TF-IDF算法提取日志文本的关键特征：

```python
def extract_features(corpus):
    """提取文本特征"""
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    # 转换文本语料为特征矩阵
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return tfidf_matrix, vectorizer.get_feature_names_out()
```

### 异常检测算法

#### 规则检测逻辑

系统通过执行预定义的SQL查询来进行规则检测：

```python
def rule_detection(db_conn, rule):
    """执行规则检测"""
    cursor = db_conn.cursor()
    
    try:
        # 执行规则SQL
        cursor.execute(rule['sql_query'])
        matches = cursor.fetchall()
        
        # 处理匹配结果
        if matches:
            anomalies = []
            for match in matches:
                anomalies.append({
                    'log_id': match['log_id'],
                    'rule_id': rule['rule_id'],
                    'anomaly_type': rule['name'],
                    'description': rule['description'],
                    'detected_by': 'rules',
                    'score': 1.0  # 规则检测确定性为100%
                })
            return anomalies
        return []
    except Exception as e:
        print(f"规则执行错误: {e}")
        return []
```

#### 暴力破解检测

针对SSH服务的暴力破解检测算法：

```python
def detect_brute_force(db_conn, threshold=5, timespan=300):
    """
    检测SSH暴力破解
    
    Args:
        db_conn: 数据库连接
        threshold: 阈值，失败尝试次数
        timespan: 时间窗口(秒)
    """
    query = """
    SELECT source_ip, user, COUNT(*) as attempts
    FROM ssh_logs
    WHERE 
        event_type = 'failed_login'
        AND timestamp >= datetime('now', '-{} seconds')
    GROUP BY source_ip, user
    HAVING attempts >= {}
    """.format(timespan, threshold)
    
    cursor = db_conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    
    anomalies = []
    for result in results:
        anomalies.append({
            'source_ip': result['source_ip'],
            'user': result['user'],
            'type': 'brute_force',
            'reason': f"检测到SSH暴力破解尝试: {result['attempts']}次失败登录"
        })
    
    return anomalies
```

#### AI异常检测 - 隔离森林算法

基于隔离森林的异常检测实现：

```python
def isolation_forest_detection(features, contamination=0.05):
    """
    使用隔离森林检测异常
    
    Args:
        features: 特征矩阵
        contamination: 预期异常占比
        
    Returns:
        异常评分列表，越高越异常
    """
    # 初始化隔离森林模型
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        random_state=42
    )
    
    # 训练模型
    model.fit(features)
    
    # 预测异常分数
    # 结果为负值，越小表示越可能是异常
    raw_scores = model.decision_function(features)
    
    # 转换为0-1范围，1表示最可能是异常
    normalized_scores = 1 - (raw_scores - min(raw_scores)) / (max(raw_scores) - min(raw_scores))
    
    return normalized_scores
```

## 4.4 核心代码结构

### 项目目录结构

```
app/
  ├── __init__.py
  ├── app.py                 # 应用入口
  ├── models/                # 数据模型
  │   ├── __init__.py
  │   ├── db.py              # 数据库定义
  │   ├── log_model.py       # 日志模型
  │   └── anomaly_model.py   # 异常模型
  ├── routes/                # API路由
  │   ├── __init__.py
  │   ├── log_routes.py      # 日志API
  │   ├── nlp_routes.py      # NLP分析API
  │   └── anomaly_routes.py  # 异常检测API
  ├── services/              # 业务服务
  │   ├── __init__.py
  │   ├── log_parser.py      # 日志解析服务
  │   ├── nlp_processor.py   # NLP处理服务
  │   └── anomaly_detector.py# 异常检测服务
  ├── static/                # 静态资源
  │   ├── css/
  │   ├── js/
  │   └── images/
  └── templates/             # 前端模板
      ├── layout.html
      ├── index.html
      └── dashboard.html
```

### 核心功能代码示例

#### 日志解析实现

app/services/log_parser.py中的核心解析功能：

```python
class LogParser:
    """日志解析器基类"""
    
    def __init__(self):
        self.parsers = {
            'ssh': SSHLogParser(),
            'web': WebLogParser(),
            'firewall': FirewallLogParser(),
            'mysql': MySQLLogParser(),
            'hdfs': HDFSLogParser(),
            'linux': LinuxLogParser()
        }
    
    def parse(self, log_data, log_type=None):
        """解析日志数据"""
        if not log_data:
            return None
            
        # 自动检测日志类型
        if log_type is None:
            log_type = self.detect_log_type(log_data)
            
        if log_type not in self.parsers:
            return {'error': f'不支持的日志类型: {log_type}'}
            
        # 使用对应的解析器处理
        return self.parsers[log_type].parse(log_data)
    
    def detect_log_type(self, log_data):
        """检测日志类型"""
        for log_type, parser in self.parsers.items():
            if parser.validate(log_data):
                return log_type
        return None
```

#### 异常检测实现

app/services/anomaly_detector.py中实现了多种异常检测器：

```python
class SSHRuleDetector:
    """SSH日志异常检测器"""
    
    def run_all_rules(self):
        """运行所有SSH检测规则"""
        anomalies = []
        anomalies.extend(self.detect_brute_force())
        anomalies.extend(self.detect_unusual_login_time())
        anomalies.extend(self.detect_multiple_countries())
        anomalies.extend(self.detect_unusual_users())
        return anomalies
    
    def detect_brute_force(self, threshold=5, timespan=300):
        """检测SSH暴力破解"""
        conn = get_db()
        cursor = conn.cursor()
        
        query = """
        SELECT source_ip, user, COUNT(*) as attempts
        FROM ssh_logs
        WHERE 
            event_type = 'failed_login'
            AND timestamp >= datetime('now', '-{} seconds')
        GROUP BY source_ip, user
        HAVING attempts >= {}
        """.format(timespan, threshold)
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        anomalies = []
        for result in results:
            anomalies.append({
                'source_ip': result['source_ip'],
                'user': result['user'],
                'type': 'brute_force',
                'reason': f"检测到SSH暴力破解尝试: {result['attempts']}次失败登录"
            })
        
        return anomalies
```

#### NLP处理实现

app/services/nlp_processor.py中的NLP处理逻辑：

```python
class NLPProcessor:
    """NLP处理引擎"""
    
    def analyze(self, text):
        """分析文本内容"""
        if not text:
            return None
            
        result = {
            'entities': self.extract_entities(text),
            'keywords': self.extract_keywords(text),
            'sentiment': self.analyze_sentiment(text)
        }
        
        return result
        
    def extract_entities(self, text):
        """提取命名实体"""
        # 实体提取逻辑
        entities = []
        # ...提取IP地址、用户名、路径等实体
        return entities
```

### 请求处理流程

app/routes/log_routes.py中的请求处理流程：

```python
@log_bp.route("/upload", methods=["POST"])
def upload_logs():
    """上传日志文件API"""
    if 'file' not in request.files:
        return jsonify({"error": "未提供文件"}), 400
        
    log_file = request.files['file']
    log_type = request.form.get('log_type')
    
    if log_file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
        
    # 读取并解析日志
    try:
        log_content = log_file.read().decode('utf-8')
        parser = LogParser()
        results = parser.parse(log_content, log_type)
        
        # 保存到数据库
        saved_logs = save_logs_to_db(results)
        
        return jsonify({
            "success": True,
            "processed": len(results),
            "saved": len(saved_logs)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
``` 