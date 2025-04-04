from flask import Blueprint, jsonify
from app.services.anomaly_detector import SSHRuleDetector,HDFSAnomalyDetector
from app.models.db import get_db

anomaly_bp = Blueprint("anomalies", __name__)

@anomaly_bp.route("/detect/ssh", methods=["GET"])
def detect_ssh_anomalies():
    detector = SSHRuleDetector()
    
    # 执行所有 SSH 检测规则
    brute_force = detector.detect_brute_force()
    user_enum = detector.detect_illegal_users()
    ddos_attack = detector.ddos()          # 新增DDoS检测
    port_scan = detector.detect_port_scanning()  # 新增端口扫描检测
    protocol_anomaly = detector.detect_protocol_anomaly()  # 新增协议异常检测
    anomalies = brute_force + user_enum+ddos_attack+port_scan+protocol_anomaly
    
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
@anomaly_bp.route("/detect/hdfs", methods=["GET"])
def detect_HDFS_anomalies():
    detector = HDFSAnomalyDetector()
    
    # 执行所有检测规则
    brute_force = detector.detect_service_brute_force()
    abnormal_deletions = detector.detect_abnormal_deletions()
    transmission_failures = detector.detect_transmission_failures()          
    anomalies = brute_force + abnormal_deletions+transmission_failures
    
    # 存储到数据库
    conn = get_db()
    cursor = conn.cursor()
    for anomaly in anomalies:
        cursor.execute("""
            INSERT INTO hdfs_anomalies 
            (timestamp, pid, anomaly_type, details)
            VALUES (datetime('now'), ?, ?, ?)
        """, (
            anomaly["pid"],
            anomaly["type"],
            anomaly["reason"]
        ))
    conn.commit()
    
    return jsonify({
        "hdfs_anomalies": anomalies,
        "count": len(anomalies)
    })
