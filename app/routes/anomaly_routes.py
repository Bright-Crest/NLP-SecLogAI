from flask import Blueprint, jsonify
from app.services.anomaly_detector import SSHRuleDetector
from app.models.db import get_db

anomaly_bp = Blueprint("anomalies", __name__)

@anomaly_bp.route("/detect", methods=["GET"])
def detect_ssh_anomalies():
    detector = SSHRuleDetector()
    
    # 执行所有 SSH 检测规则
    brute_force = detector.detect_brute_force()
    user_enum = detector.detect_illegal_users()
    
    anomalies = brute_force + user_enum
    
    # 存储到数据库
    conn = get_db()
    cursor = conn.cursor()
    for anomaly in anomalies:
        cursor.execute("""
            INSERT INTO ssh_anomalies 
            (timestamp, source_ip, anomaly_type, details)
            VALUES (datetime('now'), ?, ?, ?)
        """, (
            anomaly["source_ip"],
            anomaly["type"],
            anomaly["reason"]
        ))
    conn.commit()
    
    return jsonify({
        "ssh_anomalies": anomalies,
        "count": len(anomalies)
    })