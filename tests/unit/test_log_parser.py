#import pytest
import sys
from pathlib import Path

# 将项目根目录添加到 Python 搜索路径
root_dir = Path(__file__).parent.parent.parent  # tests/unit/ -> tests/ -> 项目根目录/
sys.path.append(str(root_dir))

# 现在可以正常导入
from app.services.log_parser import LogParser

def test_csv_parser():
    csv_content = """timestamp,ip,event,message
2023-01-01,192.168.1.1,login_failed,Invalid password"""
    logs = LogParser.parse_csv(csv_content)
    assert logs[0] == {
        "timestamp": "2023-01-01",
        "source_ip": "192.168.1.1",
        "event_type": "login_failed",
        "message": "Invalid password",
        "raw_log": "{'timestamp': '2023-01-01', 'ip': '192.168.1.1', 'event': 'login_failed', 'message': 'Invalid password'}"
    }

def test_txt_parser():
    txt_content = "2023-01-01 12:00:00 [WARN] 192.168.1.1 - Failed login"
    logs = LogParser.parse_txt(txt_content)
    assert logs[0]["event_type"] == "WARN"
    assert logs[0]["source_ip"] == "192.168.1.1"

def test_invalid_txt_log():
    logs = LogParser.parse_txt("Invalid log format")
    assert len(logs) == 0  # 应跳过无法解析的行