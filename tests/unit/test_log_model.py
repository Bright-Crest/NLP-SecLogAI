import pytest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 将项目根目录添加到 Python 搜索路径
root_dir = Path(__file__).parent.parent.parent  # tests/unit/ -> tests/ -> 项目根目录/
sys.path.append(str(root_dir))
from app.models.log_model import LogModel
from app.models.db import init_db, get_db_connection


@pytest.fixture
def setup_db():
    """测试数据库初始化夹具"""
    init_db()  # 初始化测试数据库
    yield
    # 测试后清理数据
    with get_db_connection() as conn:
        conn.cursor().execute("DELETE FROM logs")
        conn.commit()


def test_create_log_success(setup_db):
    """测试成功创建日志"""
    log_id = LogModel.create_log(
        source_ip="192.168.1.1",
        event_type="login",
        message="User logged in",
        detected_by="none"
    )
    assert isinstance(log_id, int)
    assert log_id > 0


def test_create_log_invalid_ip(setup_db):
    """测试使用无效IP创建日志"""
    with pytest.raises(ValueError, match="Invalid IP address format"):
        LogModel.create_log(
            source_ip="invalid.ip",
            event_type="login",
            message="Invalid IP test"
        )


def test_update_detection_status_success(setup_db):
    """测试成功更新检测状态"""
    # 先创建测试日志
    log_id = LogModel.create_log(
        source_ip="10.0.0.1",
        event_type="file_access",
        message="Access sensitive file"
    )

    # 更新状态
    LogModel.update_detection_status(log_id, "rules")

    # 验证更新结果
    logs = LogModel.get_logs_by_filters(source_ip="10.0.0.1")
    assert len(logs) == 1
    assert logs[0]["detected_by"] == "rules"


def test_update_detection_status_invalid(setup_db):
    """测试使用无效状态更新"""
    log_id = LogModel.create_log(
        source_ip="192.168.1.2",
        event_type="login",
        message="Test invalid status"
    )

    with pytest.raises(ValueError, match="Invalid status"):
        LogModel.update_detection_status(log_id, "invalid_status")


def test_get_logs_by_filters(setup_db):
    """测试多条件日志查询"""
    # 创建测试数据
    now = datetime.now()
    test_data = [
        {"source_ip": "192.168.1.1", "event_type": "login", "message": "Morning login",
         "timestamp": now - timedelta(hours=2)},
        {"source_ip": "192.168.1.2", "event_type": "logout", "message": "Morning logout",
         "timestamp": now - timedelta(hours=1)},
        {"source_ip": "192.168.1.1", "event_type": "login", "message": "Afternoon login",
         "timestamp": now, "detected_by": "AI"},
    ]

    # 插入测试数据（直接传入timestamp）
    for log in test_data:
        LogModel.create_log(
            source_ip=log["source_ip"],
            event_type=log["event_type"],
            message=log["message"],
            detected_by=log.get("detected_by", "none"),
            timestamp=log["timestamp"]
        )

    # 测试IP过滤
    ip_logs = LogModel.get_logs_by_filters(source_ip="192.168.1.1")
    assert len(ip_logs) == 2
    assert all(log["source_ip"] == "192.168.1.1" for log in ip_logs)

    # 测试事件类型过滤
    login_logs = LogModel.get_logs_by_filters(event_type="login")
    assert len(login_logs) == 2
    assert all(log["event_type"] == "login" for log in login_logs)

    # 测试检测状态过滤
    ai_logs = LogModel.get_logs_by_filters(detected_by="AI")
    assert len(ai_logs) == 1
    assert ai_logs[0]["detected_by"] == "AI"

    # 测试时间范围过滤
    hour_ago = now - timedelta(hours=1.5)
    recent_logs = LogModel.get_logs_by_filters(start_time=hour_ago)
    assert len(recent_logs) == 2


def test_bulk_insert_not_implemented(setup_db):
    """测试bulk_insert方法不存在时的行为"""
    test_logs = [
        {"source_ip": "10.0.0.1", "event_type": "test", "message": "bulk test 1"},
        {"source_ip": "10.0.0.2", "event_type": "test", "message": "bulk test 2"}
    ]

    # 检查是否抛出AttributeError
    with pytest.raises(AttributeError):
        LogModel.bulk_insert(test_logs)
