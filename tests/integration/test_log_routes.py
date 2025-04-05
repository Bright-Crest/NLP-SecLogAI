import pytest
import sys
from pathlib import Path

# 将项目根目录添加到 Python 搜索路径
root_dir = Path(__file__).parent.parent.parent  # tests/unit/ -> tests/ -> 项目根目录/
sys.path.append(str(root_dir))
from app.app import create_app
import io

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_upload_log(client):
    response = client.post(
        "/logs/upload",
        json={"source_ip": "192.168.1.1", "event_type": "login", "message": "test"}
    )
    assert response.status_code == 200
    assert "log_id" in response.json

def test_parse_csv_file(client):
    csv_data = "timestamp,ip,event,message\n2023-01-01,192.168.1.1,login,test"
    response = client.post(
        "/logs/parse_file",
        data={"file": (io.BytesIO(csv_data.encode()), "test.csv")},
        content_type="multipart/form-data"
    )
    assert response.status_code == 200
    assert response.json["count"] == 1
    assert response.json["logs"][0]["source_ip"] == "192.168.1.1"