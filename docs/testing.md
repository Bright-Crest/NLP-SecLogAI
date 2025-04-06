# 七、测试

本章节详细描述NLP-SecLogAI系统的测试策略、测试方法、测试用例以及测试环境配置。

## 7.1 测试策略

### 7.1.1 测试目标

- 验证系统各功能模块的正确性和稳定性
- 确保系统能够准确解析和分析各类安全日志
- 测试系统在高负载下的性能和稳定性
- 验证系统的安全性和可靠性
- 确保用户界面的易用性和响应速度

### 7.1.2 测试类型

系统测试覆盖以下主要测试类型：

| 测试类型 | 说明 | 执行阶段 |
|---------|------|---------|
| 单元测试 | 测试独立代码单元的功能正确性 | 开发阶段 |
| 集成测试 | 测试模块之间的交互 | 模块完成后 |
| 功能测试 | 验证系统功能是否符合需求 | 功能开发完成后 |
| 性能测试 | 测试系统在不同负载下的性能表现 | 系统集成后 |
| 安全测试 | 检测系统安全漏洞和脆弱性 | 系统完成后 |
| 用户接受测试 | 用户验证系统是否符合业务需求 | 发布前 |

### 7.1.3 测试环境

| 环境 | 用途 | 配置 |
|------|------|------|
| 开发环境 | 开发人员单元测试 | 本地开发机器 |
| 集成测试环境 | 持续集成和自动化测试 | 与生产环境相似但规模较小 |
| 性能测试环境 | 负载测试和性能评估 | 与生产环境规格相近 |
| 用户测试环境 | 用户验收测试 | 与生产环境配置一致 |

## 7.2 单元测试

### 7.2.1 测试框架

- Python单元测试使用**pytest**框架
- 前端单元测试使用**Jest**框架

### 7.2.2 测试覆盖率目标

- 业务核心代码覆盖率 ≥ 90%
- 整体代码覆盖率 ≥ 80%

### 7.2.3 主要测试模块

#### 日志解析器测试

```python
# test_log_parser.py
import pytest
from app.utils.log_parser import SSHLogParser, WebLogParser

def test_ssh_log_parsing():
    parser = SSHLogParser()
    log_line = "Apr 10 13:45:27 server sshd[12345]: Accepted password for user1 from 192.168.1.100 port 22 ssh2"
    result = parser.parse(log_line)
    
    assert result is not None
    assert result["timestamp"] == "Apr 10 13:45:27"
    assert result["source_ip"] == "192.168.1.100"
    assert result["user"] == "user1"
    assert result["event_type"] == "login"
    assert result["port"] == "22"

def test_web_log_parsing():
    parser = WebLogParser()
    log_line = '192.168.1.100 - - [10/Apr/2023:14:15:16 +0000] "GET /admin HTTP/1.1" 403 1234 "-" "Mozilla/5.0"'
    result = parser.parse(log_line)
    
    assert result is not None
    assert result["source_ip"] == "192.168.1.100"
    assert result["timestamp"] == "10/Apr/2023:14:15:16 +0000"
    assert result["method"] == "GET"
    assert result["path"] == "/admin"
    assert result["status_code"] == "403"
```

#### NLP处理器测试

```python
# test_nlp_processor.py
import pytest
from app.utils.nlp_processor import NLPProcessor

@pytest.fixture
def nlp_processor():
    return NLPProcessor()

def test_entity_extraction(nlp_processor):
    text = "Failed password for invalid user admin from 10.0.0.1 port 22 ssh2"
    entities = nlp_processor.extract_entities(text)
    
    assert any(e["text"] == "admin" and e["type"] == "user" for e in entities)
    assert any(e["text"] == "10.0.0.1" and e["type"] == "ip_address" for e in entities)
    assert any(e["text"] == "Failed password" and e["type"] == "auth_event" for e in entities)

def test_keyword_extraction(nlp_processor):
    text = "Failed password for invalid user admin from 10.0.0.1 port 22 ssh2"
    keywords = nlp_processor.extract_keywords(text)
    
    assert "failed" in keywords
    assert "password" in keywords
    assert "invalid" in keywords
```

#### 异常检测器测试

```python
# test_anomaly_detector.py
import pytest
from app.utils.anomaly_detector import AnomalyDetector, Rule
from app.models import Log

@pytest.fixture
def detector():
    # 创建测试规则
    rule = Rule(
        name="SSH暴力破解检测",
        sql_query="SELECT source_ip, COUNT(*) as attempts FROM logs WHERE event_type = 'failed_login' GROUP BY source_ip HAVING attempts >= 3"
    )
    return AnomalyDetector([rule])

def test_rule_based_detection(detector, mocker):
    # 模拟日志数据
    mock_logs = [
        Log(source_ip="192.168.1.100", event_type="failed_login"),
        Log(source_ip="192.168.1.100", event_type="failed_login"),
        Log(source_ip="192.168.1.100", event_type="failed_login"),
        Log(source_ip="192.168.1.101", event_type="failed_login"),
    ]
    
    # 模拟数据库查询
    mocker.patch('app.models.Log.query.filter', return_value=mock_logs)
    
    # 执行检测
    anomalies = detector.detect()
    
    assert len(anomalies) == 1
    assert anomalies[0]["source_ip"] == "192.168.1.100"
    assert anomalies[0]["count"] == 3
```

## 7.3 集成测试

### 7.3.1 API接口测试

```python
# test_log_api.py
import pytest
import json
from app import create_app
from app.models import db, Log

@pytest.fixture
def client():
    app = create_app('testing')
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
            db.drop_all()

def test_upload_log_file(client):
    """测试日志文件上传功能"""
    # 创建测试日志文件
    log_content = "Apr 10 13:45:27 server sshd[12345]: Accepted password for user1 from 192.168.1.100 port 22 ssh2"
    data = {
        'file': (io.BytesIO(log_content.encode('utf-8')), 'test.log'),
        'log_type': 'ssh'
    }
    
    response = client.post('/logs/upload', data=data, content_type='multipart/form-data')
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert result['success'] == True
    assert result['processed'] == 1
    
    # 验证日志是否保存到数据库
    log = Log.query.first()
    assert log is not None
    assert log.source_ip == "192.168.1.100"
    assert log.event_type == "login"

def test_get_logs_list(client):
    """测试获取日志列表功能"""
    # 添加测试日志数据
    log = Log(
        timestamp="2023-04-10 13:45:27",
        source_ip="192.168.1.100",
        event_type="login",
        message="Accepted password for user1 from 192.168.1.100 port 22 ssh2"
    )
    db.session.add(log)
    db.session.commit()
    
    response = client.get('/logs')
    assert response.status_code == 200
    
    result = json.loads(response.data)
    assert len(result['logs']) == 1
    assert result['logs'][0]['source_ip'] == "192.168.1.100"
    assert result['meta']['total'] == 1
```

### 7.3.2 数据流测试

```python
# test_data_flow.py
import pytest
from app import create_app
from app.utils.log_parser import LogParser
from app.utils.nlp_processor import NLPProcessor
from app.utils.anomaly_detector import AnomalyDetector
from app.models import db, Log, Anomaly

@pytest.fixture
def app_context():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

def test_end_to_end_processing(app_context):
    """测试从日志解析到异常检测的完整流程"""
    # 步骤1: 解析日志
    parser = LogParser()
    log_line = "Apr 10 13:45:27 server sshd[12345]: Failed password for invalid user admin from 10.0.0.1 port 22 ssh2"
    
    parsed_log = parser.parse(log_line)
    assert parsed_log is not None
    
    # 步骤2: 保存日志到数据库
    log = Log(
        timestamp=parsed_log["timestamp"],
        source_ip=parsed_log["source_ip"],
        event_type=parsed_log["event_type"],
        message=log_line
    )
    db.session.add(log)
    db.session.commit()
    
    # 步骤3: NLP分析
    nlp = NLPProcessor()
    entities = nlp.extract_entities(log_line)
    assert len(entities) > 0
    
    # 步骤4: 异常检测
    # 添加额外的失败日志来触发规则
    for i in range(4):
        db.session.add(Log(
            source_ip="10.0.0.1",
            event_type="failed_login",
            message=f"Failed password attempt {i}"
        ))
    db.session.commit()
    
    detector = AnomalyDetector()
    anomalies = detector.detect()
    
    assert len(anomalies) >= 1
    assert any(a["source_ip"] == "10.0.0.1" for a in anomalies)
```

## 7.4 功能测试

### 7.4.1 功能测试用例

| 测试ID | 测试目的 | 前置条件 | 测试步骤 | 预期结果 |
|-------|---------|---------|---------|---------|
| FT-001 | 验证日志上传功能 | 系统正常运行，已准备好测试日志文件 | 1. 打开日志上传页面<br>2. 选择SSH日志文件<br>3. 点击上传按钮 | 1. 系统显示上传成功消息<br>2. 数据库中存储了日志记录<br>3. 日志列表页面显示上传的记录 |
| FT-002 | 验证日志分析功能 | 系统中已有日志数据 | 1. 在日志列表中选择一条日志<br>2. 点击"分析"按钮 | 1. 系统显示分析结果页面<br>2. 页面显示实体提取结果<br>3. 页面显示关键词提取结果 |
| FT-003 | 验证异常检测功能 | 系统中已有足够触发规则的日志 | 1. 导航至异常检测页面<br>2. 点击"执行检测"按钮 | 1. 系统执行异常检测<br>2. 检测结果显示在页面上<br>3. 已检测到的异常被保存到数据库 |
| FT-004 | 验证规则管理功能 | 用户已登录并有管理权限 | 1. 导航至规则管理页面<br>2. 创建一条新规则<br>3. 修改现有规则<br>4. 删除一条规则 | 1. 新规则添加成功<br>2. 修改的规则更新成功<br>3. 规则成功删除 |
| FT-005 | 验证仪表盘功能 | 系统中已有日志和异常数据 | 1. 导航至仪表盘页面<br>2. 查看各项统计数据<br>3. 切换时间范围 | 1. 仪表盘显示正确的统计数据<br>2. 图表正确展示异常分布<br>3. 时间范围切换后数据相应更新 |

### 7.4.2 测试脚本

使用Selenium自动化功能测试：

```python
# test_ui.py
import pytest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

@pytest.fixture
def driver():
    driver = webdriver.Chrome()
    driver.maximize_window()
    yield driver
    driver.quit()

def test_log_upload(driver):
    # 导航到上传页面
    driver.get("http://localhost:5000/logs/upload")
    
    # 定位文件上传输入框并上传文件
    file_input = driver.find_element(By.ID, "log-file")
    file_input.send_keys("/path/to/test/ssh.log")
    
    # 选择日志类型
    log_type_select = driver.find_element(By.ID, "log-type")
    log_type_select.select_by_visible_text("SSH")
    
    # 点击上传按钮
    upload_button = driver.find_element(By.ID, "upload-button")
    upload_button.click()
    
    # 等待上传完成并验证结果
    success_message = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "alert-success"))
    )
    
    assert "成功上传" in success_message.text
    
    # 导航到日志列表页面验证记录
    driver.get("http://localhost:5000/logs")
    log_entries = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    assert len(log_entries) > 0
```

## 7.5 性能测试

### 7.5.1 性能测试指标

| 测试指标 | 目标值 | 临界值 |
|---------|-------|-------|
| 页面加载时间 | ≤ 2秒 | ≤ 3秒 |
| 单日志分析响应时间 | ≤ 1秒 | ≤ 2秒 |
| 批量日志处理速度 | ≥ 1000条/分钟 | ≥ 500条/分钟 |
| 异常检测响应时间 | ≤ 5秒(10000条日志) | ≤ 10秒 |
| 并发用户支持 | ≥ 50 | ≥ 30 |
| 数据库查询响应时间 | ≤ 0.5秒 | ≤ 1秒 |

### 7.5.2 负载测试

使用Locust进行负载测试：

```python
# locustfile.py
from locust import HttpUser, task, between

class SecLogAIUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def get_logs(self):
        self.client.get("/logs", params={"page": 1, "limit": 50})
    
    @task(1)
    def get_dashboard(self):
        self.client.get("/api/dashboard")
    
    @task(2)
    def analyze_log(self):
        self.client.post("/nlp/analyze", json={
            "text": "Failed password for invalid user admin from 10.0.0.1 port 22 ssh2",
            "analysis_type": "full"
        })
    
    @task(1)
    def detect_anomalies(self):
        self.client.get("/anomalies/detect/all")
```

### 7.5.3 性能测试报告

性能测试将生成以下报告内容：

1. 响应时间分布
2. 吞吐量统计
3. 错误率统计
4. 资源利用率(CPU、内存、网络、磁盘IO)
5. 数据库性能指标
6. 瓶颈分析和优化建议

## 7.6 安全测试

### 7.6.1 安全测试范围

| 测试类型 | 说明 | 工具 |
|---------|------|------|
| 漏洞扫描 | 检测已知安全漏洞 | OWASP ZAP, Nessus |
| 代码安全审计 | 分析代码中的安全问题 | Bandit, SonarQube |
| 渗透测试 | 模拟黑客攻击场景 | Metasploit, Burp Suite |
| 安全配置评估 | 评估服务器和应用配置安全性 | CIS-CAT, OpenSCAP |

### 7.6.2 测试用例

| 测试ID | 测试目的 | 测试步骤 | 预期结果 |
|-------|---------|---------|---------|
| ST-001 | SQL注入测试 | 1. 在查询参数中注入SQL语句<br>2. 在表单字段中注入SQL语句 | 系统应正确处理输入，不执行注入的SQL语句 |
| ST-002 | XSS攻击测试 | 在日志文本和规则名称中输入JavaScript代码 | 系统应转义或过滤特殊字符，不执行JavaScript代码 |
| ST-003 | 认证绕过测试 | 尝试未经认证访问受保护资源 | 系统应拒绝访问并重定向至登录页面 |
| ST-004 | CSRF测试 | 尝试从外部站点提交表单到系统 | 系统应验证CSRF令牌，拒绝未授权请求 |
| ST-005 | 文件上传测试 | 上传恶意文件或非预期类型文件 | 系统应验证文件类型和内容，拒绝恶意文件 |

## 7.7 用户验收测试

### 7.7.1 用户验收测试场景

| 测试ID | 测试场景 | 测试步骤 | 验收标准 |
|-------|---------|---------|---------|
| UAT-001 | 日志解析与识别 | 使用实际环境的各类日志进行上传和处理 | 系统能准确解析各类安全日志，识别率≥95% |
| UAT-002 | 安全事件检测 | 导入含有已知安全事件的日志数据 | 系统能够检测出已知的安全事件，检出率≥90% |
| UAT-003 | 误报率评估 | 导入正常行为的日志数据 | 系统误报率≤5% |
| UAT-004 | 大规模数据处理 | 导入大量(>100,000条)日志记录 | 系统性能保持稳定，处理时间合理 |
| UAT-005 | 综合业务场景 | 模拟实际安全运维工作流程 | 系统满足用户工作需求，提升工作效率 |

### 7.7.2 用户反馈收集

用户验收测试过程中将收集以下反馈：

1. 系统易用性评价
2. 功能完整性评价
3. 性能满意度评价
4. 发现的问题和改进建议
5. 总体满意度评分

## 7.8 测试报告

### 7.8.1 报告模板

每轮测试完成后，将生成包含以下内容的测试报告：

1. 测试概述
   - 测试范围
   - 测试环境
   - 测试团队

2. 测试结果摘要
   - 执行的测试用例数量
   - 通过/失败用例统计
   - 发现的关键问题

3. 详细测试结果
   - 按测试类型分类的结果
   - 失败用例的详细描述
   - 附带的证据(截图、日志等)

4. 问题汇总
   - 按严重程度分类的问题列表
   - 问题状态跟踪
   - 解决方案建议

5. 性能测试数据
   - 响应时间统计
   - 资源利用率图表
   - 与基准的对比

6. 结论和建议
   - 测试结论
   - 风险评估
   - 改进建议

### 7.8.2 缺陷跟踪流程

1. 缺陷发现与记录
2. 缺陷分类与优先级评估
3. 缺陷分配与修复
4. 修复验证
5. 缺陷关闭

## 7.9 持续集成测试

### 7.9.1 CI/CD流程

```mermaid
graph TD
    A[代码提交] --> B[自动化构建]
    B --> C[单元测试]
    C --> D{测试通过?}
    D -->|是| E[集成测试]
    D -->|否| A
    E --> F{测试通过?}
    F -->|是| G[部署到测试环境]
    F -->|否| A
    G --> H[功能测试]
    H --> I{测试通过?}
    I -->|是| J[部署到生产环境]
    I -->|否| A
```

### 7.9.2 GitHub Actions配置

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:latest
        env:
          POSTGRES_USER: seclogai_test
          POSTGRES_PASSWORD: seclogai_test
          POSTGRES_DB: seclogai_test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest pytest-cov
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test with pytest
      run: |
        pytest --cov=app tests/
      env:
        DATABASE_URL: postgresql://seclogai_test:seclogai_test@localhost:5432/seclogai_test
        FLASK_ENV: testing
        
    - name: Upload coverage report
      uses: codecov/codecov-action@v1
``` 