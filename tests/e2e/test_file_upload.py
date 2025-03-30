import requests
import os

BASE_URL = "http://localhost:5000"


def test_file_upload_flow():
    # 1. 上传CSV文件
    with open("test_logs.csv", "w") as f:
        f.write("timestamp,ip,event,message\n2023-01-01,192.168.1.1,login,test")

    with open("test_logs.csv", "rb") as f:
        response = requests.post(
            f"{BASE_URL}/logs/parse_file",
            files={"file": ("test_logs.csv", f)}
        )
    assert response.status_code == 200
    log_id = response.json()["logs"][0].get("log_id")

    # 2. 验证数据库插入
    db_response = requests.post(
        f"{BASE_URL}/logs/query",
        json={"filters": {"source_ip": "192.168.1.1"}}
    )
    assert db_response.status_code == 200
    assert len(db_response.json()["logs"]) >= 1

    # 清理
    os.remove("test_logs.csv")