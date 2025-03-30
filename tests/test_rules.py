from datetime import datetime, timedelta
import pytest
from app.models.db import get_db, init_db
from app.services.anomaly_detector import SSHRuleDetector

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
