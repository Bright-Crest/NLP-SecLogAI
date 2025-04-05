from flask import Blueprint, jsonify
from app.services.anomaly_detector import SSHRuleDetector,WebLogDetector,FirewallDetector,MySQLDetector
from app.models.db import get_db

anomaly_bp = Blueprint("anomalies", __name__)

@anomaly_bp.route("/detect/ssh", methods=["GET"])
def detect_ssh_anomalies():
    detector = SSHRuleDetector()
    
    # 执行所有 SSH 检测规则
    anomalies = detector.run_all_rules()
    
    # 存储到数据库
    conn = get_db()
    cursor = conn.cursor()
    for anomaly in anomalies:
        cursor.execute("""
            INSERT INTO ssh_anomalies 
            (timestamp, source_ip, user, anomaly_type, details, country_codes)
            VALUES (datetime('now'), ?, ?, ?, ?, ?)
        """, (
            anomaly.get("source_ip"),
            anomaly.get("user"),
            anomaly["type"],
            anomaly["reason"],
            ",".join(anomaly.get("countries", []))
        ))
    conn.commit()
    
    return jsonify({
        "ssh_anomalies": anomalies,
        "count": len(anomalies)
    })

@anomaly_bp.route("/detect/web", methods=["GET"])
def detect_web_anomalies():
    detector = WebLogDetector()
    anomalies = (
        detector.detect_sqli() +
        detector.detect_xss() +
        detector.detect_brute_force() +
        detector.detect_scanner() +
        detector.detect_directory_traversal() +      
        detector.detect_sensitive_files() +           
        detector.detect_suspicious_ua() +             
        detector.detect_high_frequency_requests() +   
        detector.detect_abnormal_methods()            
    )
    
    # 存储到web_anomalies表
    conn = get_db()
    cursor = conn.cursor()
    for anomaly in anomalies:
        cursor.execute("""
            INSERT INTO web_anomalies 
            (source_ip, anomaly_type, details)
            VALUES (?, ?, ?)
        """, (
            anomaly["source_ip"],
            anomaly["type"],
            anomaly["reason"]
        ))
    conn.commit()
    return jsonify({"web_anomalies": anomalies})


@anomaly_bp.route("/detect/firewall", methods=["GET"])
def detect_firewall_anomalies():
    detector = FirewallDetector()
    anomalies = (
        detector.detect_port_scan() +
        detector.detect_ddos() +
        detector.detect_abnormal_protocols() +
        detector.detect_internal_outbound()
    )
    
    # 存储到firewall_anomalies表
    cursor = get_db().cursor()
    for anomaly in anomalies:
        cursor.execute("""
            INSERT INTO firewall_anomalies 
            (source_ip, anomaly_type, details)
            VALUES (?, ?, ?)
        """, (
            anomaly["source_ip"],
            anomaly["type"],
            anomaly["reason"]
        ))
    return jsonify({"firewall_anomalies": anomalies})


@anomaly_bp.route("/detect/mysql", methods=["GET"])
def detect_mysql_anomalies():
    detector = MySQLDetector()
    anomalies = (
        detector.detect_brute_force() +
        detector.detect_high_risk_operations() +
        detector.detect_privilege_escalation() +
        detector.detect_data_exfiltration() +
        detector.detect_off_hour_access()
    )
    
    # 存储到mysql_anomalies表
    cursor = get_db().cursor()
    for anomaly in anomalies:
        cursor.execute("""
            INSERT INTO mysql_anomalies 
            (user, source_ip, anomaly_type, details)
            VALUES (?, ?, ?, ?)
        """, (
            anomaly.get("user"),
            anomaly.get("source_ip"),
            anomaly["type"],
            anomaly["reason"]
        ))
    return jsonify({"mysql_anomalies": anomalies})
