from datetime import datetime, timedelta
import pytest
from app.models.db import get_db, init_db
from app.services.anomaly_detector import SSHRuleDetector,HDFSAnomalyDetector
import sys
sys.path.append(r'E:\NLP-SecLogAI')
print(sys.path)
@pytest.fixture
def app():
    # 创建测试用 Flask 应用
    from app.app import app
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:"
    })
    
    with app.app_context():
        init_db()  # 初始化内存数据库
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def db_session(app):
    # 提供干净的数据库会话
    conn = get_db()
    yield conn
    # 测试后清理数据库
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ssh_logs")
    cursor.execute("DELETE FROM ssh_anomalies")
    conn.commit()

def test_brute_force_detection(db_session):
    # 暴力破解检测测试
    cursor = db_session.cursor()
    
    # 插入6次失败登录（超过阈值5次）
    for i in range(6):
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),
            "authentication_failure",
            "192.168.1.100"
        ))
    db_session.commit()
    
    # 执行检测
    detector = SSHRuleDetector()
    results = detector.detect_brute_force()
    
    assert len(results) == 1
    assert results[0]["source_ip"] == "192.168.1.100"
    assert "暴力破解" in results[0]["reason"]

def test_user_enumeration_detection(db_session):
    # 用户枚举检测测试
    cursor = db_session.cursor()
    
    # 插入4个不同用户（超过阈值3次）
    for user in ["alice", "bob", "charlie", "david"]:
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip, user_name)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now(),
            "invalid_user",
            "192.168.1.101",
            user
        ))
    db_session.commit()
    
    detector = SSHRuleDetector()
    results = detector.detect_illegal_users()
    
    assert len(results) == 1
    assert results[0]["source_ip"] == "192.168.1.101"
    assert "用户枚举" in results[0]["reason"]

def test_ddos_detection(db_session):
    # DDoS检测测试（超过60次无效登录/分钟）
    cursor = db_session.cursor()
    
    # 插入65次无效登录（超过阈值60）
    for i in range(65):
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(seconds=30),  # 最近30秒内
            "invalid_user",
            f"192.168.1.{i%5}"  # 分散到不同IP
        ))
    db_session.commit()
    
    detector = SSHRuleDetector()
    results = detector.ddos()
    
    assert len(results) == 1
    assert results[0]["type"] == "global_ddos_attack"
    assert "超过60次" in results[0]["reason"]

def test_no_ddos_detection(db_session):
    # 不触发DDoS检测测试（刚好60次）
    cursor = db_session.cursor()
    
    # 插入60次无效登录（等于阈值）
    for i in range(60):
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=1),
            "invalid_user",
            f"192.168.1.{i%5}"
        ))
    db_session.commit()
    
    detector = SSHRuleDetector()
    results = detector.ddos()
    
    assert len(results) == 0

def test_port_scanning_detection(db_session):
    # 端口扫描检测测试（超过3个不同端口）
    cursor = db_session.cursor()
    
    # 同一IP尝试4个不同端口
    ports = [22, 2222, 8080, 3389]
    for port in ports:
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip, port)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=3),
            "connection_attempt",
            "192.168.1.200",
            port
        ))
    db_session.commit()
    
    detector = SSHRuleDetector()
    results = detector.detect_port_scanning()
    
    assert len(results) == 1
    assert results[0]["source_ip"] == "192.168.1.200"
    assert results[0]["ports"] == 4
    assert "端口扫描" in results[0]["reason"]

def test_protocol_anomaly_detection(db_session):
    # 协议异常检测测试（非22端口）
    cursor = db_session.cursor()
    
    # 插入3次非22端口连接尝试
    for i in range(3):
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip, port)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=30),
            "connection_attempt",
            "192.168.1.250",
            2222  # 非常用端口
        ))
    db_session.commit()
    
    detector = SSHRuleDetector()
    results = detector.detect_protocol_anomaly()
    
    assert len(results) == 1
    assert results[0]["port"] == 2222
    assert results[0]["attempts"] == 3
    assert "非常用SSH端口" in results[0]["reason"]

def test_no_protocol_anomaly_detection(db_session):
    # 不触发协议异常检测测试（使用22端口）
    cursor = db_session.cursor()
    
    # 插入3次22端口连接尝试
    for i in range(3):
        cursor.execute("""
            INSERT INTO ssh_logs 
            (timestamp, event_type, source_ip, port)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=30),
            "connection_attempt",
            "192.168.1.250",
            22  # 标准端口
        ))
    db_session.commit()
    
    detector = SSHRuleDetector()
    results = detector.detect_protocol_anomaly()
    
    assert len(results) == 0


def test_service_brute_force_detection(db_session):
    # 组件级暴力操作检测测试（超过3次）
    cursor = db_session.cursor()
    
    # 插入4次同一组件的E3事件（超过阈值3次）
    component = "DataNode"
    pid = 1234
    for i in range(4):
        cursor.execute("""
            INSERT INTO hdfs_logs 
            (timestamp, pid, component, Eventid)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),  # 4分钟内
            pid,
            component,
            "E3"
        ))
    db_session.commit()
    
    detector = HDFSAnomalyDetector()
    results = detector.detect_service_brute_force()
    
    assert len(results) == 1
    assert results[0]["type"] == "service_brute_force"
    assert results[0]["pid"] == pid
    assert f"组件{component}异常4次" in results[0]["reason"]

def test_normal_service_operation(db_session):
    # 正常服务操作测试（未超过阈值）
    cursor = db_session.cursor()
    
    # 插入3次同一组件的E3事件（等于阈值）
    component = "NameNode"
    pid = 5678
    for i in range(3):
        cursor.execute("""
            INSERT INTO hdfs_logs 
            (timestamp, pid, component, Eventid)
            VALUES (?, ?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),
            pid,
            component,
            "E3"
        ))
    db_session.commit()
    
    detector = HDFSAnomalyDetector()
    results = detector.detect_service_brute_force()
    
    assert len(results) == 0

def test_abnormal_deletions_detection(db_session):
    # 异常删除行为检测测试（超过5次）
    cursor = db_session.cursor()
    
    # 插入6次删除操作（超过阈值5次）
    pid = 9012
    for i in range(6):
        cursor.execute("""
            INSERT INTO hdfs_logs 
            (timestamp, pid, Eventid)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),
            pid,
            "E4"  # 也可以是E8
        ))
    db_session.commit()
    
    detector = HDFSAnomalyDetector()
    results = detector.detect_abnormal_deletions()
    
    assert len(results) == 1
    assert results[0]["type"] == "abnormal_deletions"
    assert results[0]["pid"] == pid
    assert f"进程{pid}5分钟内删除6次" in results[0]["reason"]

def test_normal_deletions(db_session):
    # 正常删除行为测试（未超过阈值）
    cursor = db_session.cursor()
    
    # 插入5次删除操作（等于阈值）
    pid = 3456
    for i in range(5):
        cursor.execute("""
            INSERT INTO hdfs_logs 
            (timestamp, pid, Eventid)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),
            pid,
            "E8"
        ))
    db_session.commit()
    
    detector = HDFSAnomalyDetector()
    results = detector.detect_abnormal_deletions()
    
    assert len(results) == 0

def test_transmission_failures_storm(db_session):
    # 传输失败风暴检测测试（超过5次）
    cursor = db_session.cursor()
    
    # 插入6次传输失败（超过阈值5次）
    pid = 7890
    for i in range(6):
        cursor.execute("""
            INSERT INTO hdfs_logs 
            (timestamp, pid, Eventid)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),
            pid,
            "E10"
        ))
    db_session.commit()
    
    detector = HDFSAnomalyDetector()
    results = detector.detect_transmission_failures()
    
    assert len(results) == 1
    assert results[0]["type"] == "transmission_failures"
    assert results[0]["pid"] == pid
    assert f"传输失败：5分钟内传输失败6次" in results[0]["reason"]

def test_normal_transmission(db_session):
    # 正常传输测试（未超过阈值）
    cursor = db_session.cursor()
    
    # 插入5次传输失败（等于阈值）
    pid = 1357
    for i in range(5):
        cursor.execute("""
            INSERT INTO hdfs_logs 
            (timestamp, pid, Eventid)
            VALUES (?, ?, ?)
        """, (
            datetime.now() - timedelta(minutes=4),
            pid,
            "E10"
        ))
    db_session.commit()
    
    detector = HDFSAnomalyDetector()
    results = detector.detect_transmission_failures()
    
    assert len(results) == 0