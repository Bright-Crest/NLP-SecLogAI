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

# import pytest
# from datetime import datetime, timedelta
# from app.models.db import get_db

# @pytest.fixture
# def client():
#     # 创建测试客户端
#     from app.app import app
#     app.config['TESTING'] = True
#     with app.test_client() as client:
#         with app.app_context():
#             # 初始化测试数据库
#             conn = get_db()
#             cursor = conn.cursor()
#             cursor.execute("DELETE FROM logs")
#             cursor.execute("DELETE FROM anomalies")
#             conn.commit()
#         yield client

# def test_brute_force_detection(client):
#     """测试暴力破解检测规则"""
#     # 插入15次失败登录日志
#     for i in range(15):
#         client.post('/logs/upload', json={
#             "log_content": f"{(datetime.now()-timedelta(minutes=4)).strftime('%Y-%m-%d %H:%M:%S')} - attacker - login_failed - 192.168.1.100"
#         })
    
#     # 执行检测
#     response = client.get('/anomalies/detect')
    
#     # 验证检测结果
#     assert response.status_code == 200
#     anomalies = response.json['anomalies']
#     assert any(a['reason'].startswith("暴力破解尝试") for a in anomalies), "应检测到暴力破解"

# def test_abnormal_ip_detection(client):
#     """测试异常IP检测规则"""
#     # 插入同一用户不同IP的登录日志
#     for ip in ['192.168.1.101', '192.168.1.102', '192.168.1.103', '192.168.1.104']:
#         client.post('/logs/upload', json={
#             "log_content": f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - vip_user - login_success - {ip}"
#         })
    
#     response = client.get('/anomalies/detect')
#     assert any(a['reason'].startswith("异常IP登录") for a in response.json['anomalies']), "应检测到异常IP"

# def test_keyword_detection(client):
#     """测试敏感关键词检测规则"""
#     # 插入包含敏感词的日志
#     client.post('/logs/upload', json={
#         "log_content": "2023-10-01 12:00:00 - admin - sudo privilege escalation - 10.0.0.1"
#     })
    
#     response = client.get('/anomalies/detect')
#     assert any("高危关键词：sudo" in a['reason'] for a in response.json['anomalies']), "应检测到敏感词"

# def test_upload_log(client):
#     response = client.post("/logs/upload", json={"log_content": "2025-03-24 12:34:56 - User 'admin' login failed from 192.168.1.100"})
#     assert response.status_code == 200
#     assert response.json["status"] == "success"