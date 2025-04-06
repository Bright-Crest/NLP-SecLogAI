from flask import Blueprint, jsonify, request, render_template
from app.services.anomaly_detector import SSHRuleDetector,WebLogDetector,FirewallDetector,MySQLDetector,HDFSAnomalyDetector
from app.models.db import get_db

anomaly_bp = Blueprint("anomalies", __name__)


@anomaly_bp.route('/')
def index():
    return render_template('anomaly_index.html')


@anomaly_bp.route('/stats')
def get_statistics():
    # log_type = request.args.get('type',"none")
    # if log_type != "none":
    #     # 示例数据结构，需替换为真实数据库查询
    #     return jsonify({
    #         "attack_types": [
    #             {"type": "暴力破解", "count": 15},
    #             {"type": "SQL注入", "count": 8}
    #         ],
    #         "time_series": [
    #             {"hour": "00:00", "count": 3},
    #             {"hour": "12:00", "count": 12}
    #         ]
    #     })
    log_type = request.args.get('type')
    conn = get_db()
    cursor = conn.cursor()

    # 获取攻击类型分布（从对应异常表按type字段分组）
    cursor.execute(f"""
        SELECT anomaly_type as type, COUNT(*) as count 
        FROM {log_type}_anomalies 
        GROUP BY anomaly_type
    """)
    attack_types = [dict(row) for row in cursor.fetchall()]

    # 获取时间趋势（按小时聚合）
    cursor.execute(f"""
        SELECT 
            strftime('%H', timestamp) as hour, 
            COUNT(*) as count 
        FROM {log_type}_anomalies 
        GROUP BY hour
        ORDER BY hour
    """)
    time_series = [{'hour': row['hour'], 'count': row['count']} for row in cursor.fetchall()]

    return jsonify({
        "attack_types": attack_types,
        "time_series": time_series
    })



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
    conn.close()
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
    conn.close()
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
    conn = get_db()
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()
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
    conn = get_db()
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()
    return jsonify({"mysql_anomalies": anomalies})


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
    conn.close()
    return jsonify({
        "hdfs_anomalies": anomalies,
        "count": len(anomalies)
    })