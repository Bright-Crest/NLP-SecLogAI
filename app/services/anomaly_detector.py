from app.models.db import get_db
from datetime import datetime, timedelta

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
