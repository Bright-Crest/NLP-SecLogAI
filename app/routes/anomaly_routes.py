# from flask import Blueprint, jsonify
# from app.models.db import get_db
# from app.services.anomaly_detector import RuleBasedDetector

# anomaly_bp = Blueprint("anomalies", __name__)

# @anomaly_bp.route("/detect", methods=["GET"])
# def detect_anomalies():
#     detector = RuleBasedDetector()
#     results = detector.run_all_rules()
    
#     # 存储到 anomalies 表
#     conn = get_db()
#     cursor = conn.cursor()
#     for anomaly in results:
#         cursor.execute("""
#             INSERT INTO anomalies 
#             (timestamp, username, event, ip_address, reason, detected_by)
#             VALUES (datetime('now'), ?, ?, ?, ?, 'rules')
#         """, (
#             anomaly.get("username"),
#             anomaly.get("type"),
#             anomaly.get("ip_address"),
#             anomaly.get("reason")
#         ))
#     conn.commit()
    
#     return jsonify({"anomalies": results})

#     # 规则：同一 IP 5 分钟内失败登录超过 10 次
#     cursor.execute("""
#     SELECT username, ip_address, COUNT(*) AS attempts
#     FROM logs WHERE event='login_failed'
#     GROUP BY username, ip_address HAVING attempts > 10
#     """)
    
#     anomalies = cursor.fetchall()
#     results = []

#     for anomaly in anomalies:
#         results.append({
#             "username": anomaly["username"],
#             "ip_address": anomaly["ip_address"],
#             "reason": "Possible brute-force attack ({} failed attempts)".format(anomaly["attempts"])
#         })

#     return jsonify({"anomalies": results})


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