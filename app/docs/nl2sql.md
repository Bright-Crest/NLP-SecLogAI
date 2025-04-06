# 基于NLP的安全日志分析与异常检测系统

这是一个创新型安全日志分析系统，利用自然语言处理技术对安全日志进行智能化处理。本系统突破了传统SIEM系统仅依赖规则匹配的局限，实现了对日志的语义理解和智能化处理。

## 功能特点

- **NLP日志解析**：使用自然语言处理技术将非结构化日志转换为结构化数据
- **智能查询引擎**：通过自然语言输入直接查询日志，无需编写复杂SQL
- **异常检测**：结合规则和机器学习，自动识别异常行为和攻击模式
- **Web界面**：直观的用户界面，方便查询和可视化日志数据

## 项目结构

```
log_analysis_project/
│── app/                          # 核心应用目录
│   ├── static/                    # 静态文件（CSS, JS）
│   │   ├── css/
│   │   ├── js/
│   ├── templates/                 # 前端HTML页面
│   │   ├── index.html              # 日志查询主界面
│   ├── routes/                    # API路由
│   │   ├── __init__.py             # 路由包初始化
│   │   ├── nlp_routes.py           # NLP查询API
│   ├── models/                    # 数据库模型
│   │   ├── db.py                   # 数据库连接
│   ├── services/                  # 业务逻辑
│   │   ├── nlp_processor.py        # NLP处理模块
│   ├── app.py                     # Flask应用入口
│── tests/                        # 测试文件
│── .env.example                  # 环境变量示例
│── requirements.txt              # 依赖项
│── run.py                        # 启动脚本
│── README.md                     # 项目文档
```

## 安装与配置

1. 克隆项目
```bash
git clone https://github.com/your-username/log-analysis-project.git
cd log_analysis_project
```

2. 创建虚拟环境
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
# 复制示例文件
cp .env.example .env
# 编辑.env文件，设置OpenRouter API密钥
```

## 运行应用

```bash
python run.py
```

应用将在 http://localhost:5000 运行。

## NLP查询功能详解

本系统通过自然语言处理技术将用户的自然语言查询转换为结构化SQL查询，无需掌握SQL语法即可快速查询日志数据。

### 支持的查询类型

项目支持以下类型的自然语言查询：

1. **时间范围查询**：
   - "最近24小时的所有日志"
   - "过去1周内的安全事件"
   - "今天的登录记录"
   - "昨天的异常活动"
   - "从2023-01-01到2023-01-31的日志"

2. **用户行为查询**：
   - "admin用户的登录失败记录"
   - "最近24小时内哪些用户尝试登录超过5次"
   - "用户root的所有操作"

3. **异常检测查询**：
   - "显示所有异常登录尝试"
   - "今天发生的可疑活动"
   - "检测到的暴力破解攻击"
   - "被AI识别的异常行为"

4. **规则相关查询**：
   - "查看所有检测规则"
   - "规则触发的异常"

5. **IP地址相关查询**：
   - "来自IP 192.168.1.100的所有活动"
   - "多次失败登录的IP地址"

### 查询预处理功能

系统具备强大的查询预处理能力：

1. **时间表达式识别**：
   - 相对时间：最近X小时/天/周/月
   - 特定时间点：今天、昨天、本周、上周
   - 时间范围：从X到Y的时间段

2. **实体识别**：
   - 用户名识别和规范化
   - IP地址识别
   - 事件类型匹配（登录、注销、访问等）
   - 状态识别（成功、失败、拒绝等）

3. **查询意图分析**：
   - 自动判断是查询日志、异常还是规则
   - 智能选择需要的表和字段

4. **多表关联**：
   - 自动生成JOIN查询关联多个表
   - 处理复杂的查询逻辑

### 查询示例

#### 基本查询示例
```
用户查询: "最近24小时admin登录失败次数"

生成SQL: 
SELECT COUNT(*) as failure_count 
FROM logs 
WHERE timestamp >= datetime('now', '-24 hour') 
AND event_type = 'login' 
AND message LIKE '%failure%' 
AND user = 'admin'
```

#### 复杂查询示例
```
用户查询: "上周被AI检测出的异常行为中涉及到root用户的事件"

生成SQL:
SELECT a.anomaly_id, a.detected_by, a.anomaly_type, l.timestamp, l.source_ip, l.event_type, l.message
FROM anomalies a
JOIN logs l ON a.log_id = l.log_id
WHERE l.timestamp >= datetime('now', '-14 day')
AND l.timestamp <= datetime('now', '-7 day')
AND a.detected_by IN ('AI', 'both')
AND l.user = 'root'
ORDER BY l.timestamp DESC
```

## API接口

### 自然语言转SQL

```
POST /nlp/query
Content-Type: application/json

{
  "query": "最近24小时admin登录失败次数"
}
```

响应：
```json
{
  "sql": "SELECT COUNT(*) as failure_count FROM logs WHERE timestamp >= datetime('now', '-24 hour') AND status = 'failure' AND event_type = 'logon' AND user = 'admin'",
  "results": [{"failure_count": 5}],
  "original_query": "最近24小时admin登录失败次数"
}
```

## 技术栈

- **后端**：Flask (Python)
- **数据库**：SQLite
- **NLP引擎**：OpenRouter API (Claude-3-Opus)
- **前端**：HTML, JavaScript

## 贡献

欢迎贡献代码或提出问题！请提交Pull Request或创建Issue。 