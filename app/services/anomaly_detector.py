from app.models.db import get_db
from datetime import datetime, timedelta
import ipaddress
import requests  # 用于获取IP地理位置（可选）

class SSHRuleDetector:
    def __init__(self):
        self.conn = get_db()
    
    def detect_brute_force(self):
        """检测SSH暴力破解（5分钟内同一IP超过5次失败）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(*) as attempts 
            FROM ssh_logs 
            WHERE event_type IN ('authentication_failure', 'invalid_user')
              AND timestamp >= ?
            GROUP BY source_ip 
            HAVING attempts > 5
        """, (datetime.now() - timedelta(minutes=5),))
        
        return [{
            "type": "ssh_brute_force",
            "source_ip": row["source_ip"],
            "attempts": row["attempts"],
            "reason": f"SSH暴力破解检测：{row['attempts']}次失败尝试"
        } for row in cursor.fetchall()]

    def detect_illegal_users(self):
        """检测可疑用户枚举（同一IP尝试超过3个不同用户）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(DISTINCT user_name) as user_count
            FROM ssh_logs 
            WHERE event_type = 'invalid_user'
              AND timestamp >= ?
            GROUP BY source_ip 
            HAVING user_count > 3
        """, (datetime.now() - timedelta(hours=1),))
        
        return [{
            "type": "user_enumeration",
            "source_ip": row["source_ip"],
            "users": row["user_count"],
            "reason": f"可疑用户枚举：尝试了{row['user_count']}个不同用户"
        } for row in cursor.fetchall()]

# class RuleBasedDetector:
#     def __init__(self):
#         self.conn = get_db()
    
#     def detect_brute_force(self, time_window=5, max_attempts=10):
#         """
#         暴力破解检测：同一IP在指定时间内失败登录超过阈值
#         """
#         cursor = self.conn.cursor()
#         query = f"""
#             SELECT username, ip_address, COUNT(*) as attempts 
#             FROM logs 
#             WHERE event = 'login_failed' 
#               AND timestamp >= datetime('now', '-{time_window} minutes')
#             GROUP BY username, ip_address 
#             HAVING attempts > {max_attempts}
#         """
#         cursor.execute(query)
#         return [
#             {
#                 "type": "brute_force",
#                 "username": row["username"],
#                 "ip_address": row["ip_address"],
#                 "attempts": row["attempts"],
#                 "reason": f"暴力破解尝试：{row['attempts']}次失败登录"
#             } 
#             for row in cursor.fetchall()
#         ]

#     def detect_abnormal_ip(self, user, max_ips=3):
#         """
#         异常IP检测：用户24小时内使用不同IP数量超过阈值
#         """
#         cursor = self.conn.cursor()
#         query = f"""
#             SELECT COUNT(DISTINCT ip_address) as ip_count 
#             FROM logs 
#             WHERE username = '{user}' 
#               AND timestamp >= datetime('now', '-1 day')
#         """
#         cursor.execute(query)
#         result = cursor.fetchone()
#         if result and result["ip_count"] > max_ips:
#             return {
#                 "type": "abnormal_ip",
#                 "username": user,
#                 "reason": f"异常IP登录：24小时内使用{result['ip_count']}个不同IP"
#             }
#         return None

#     def detect_keywords(self, keywords=["root", "sudo", "privilege"]):
#         """
#         关键字检测：日志中包含高危关键词
#         """
#         cursor = self.conn.cursor()
#         placeholders = ", ".join(f"'{k}'" for k in keywords)
#         query = f"""
#             SELECT * FROM logs 
#             WHERE LOWER(event) IN ({placeholders})
#         """
#         cursor.execute(query)
#         return [
#             {
#                 "type": "sensitive_keyword",
#                 "log_id": row["id"],
#                 "event": row["event"],
#                 "reason": f"检测到高危关键词：{row['event']}"
#             }
#             for row in cursor.fetchall()
#         ]

#     def run_all_rules(self):
#         """执行所有规则检测"""
#         anomalies = []
#         anomalies.extend(self.detect_brute_force())
#         anomalies.extend(self.detect_keywords())
#         # 其他规则可在此扩展
#         return anomalies