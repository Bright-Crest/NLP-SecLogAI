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
    
    def ddos(self):
        """检测DDoS行为（全局1分钟内无效登录超过60次）"""
        cursor = self.conn.cursor()   
        # 执行SQL查询
        cursor.execute("""
            SELECT COUNT(*) as total_invalid_attempts
            FROM ssh_logs
            WHERE event_type = 'invalid_user'  
            AND timestamp >= ?
        """, (datetime.now() - timedelta(minutes=1),))
    
        # 获取查询结果
        result = cursor.fetchone()
    
        # 判断是否超过阈值
        if result and result["total_invalid_attempts"] > 60:
            return [{
                "type": "global_ddos_attack",
                "source_ip": "null",
                "reason": f"全局DDoS攻击迹象：1分钟内无效登录尝试{result['total_invalid_attempts']}次（超过60次阈值）"
            }]
        else:
            return []
    
    def detect_port_scanning(self):
        """检测端口扫描行为（同一IP尝试连接超过3个不同端口）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(DISTINCT port) as port_count
            FROM ssh_logs 
            WHERE event_type IN ('connection_attempt', 'invalid_port')
              AND timestamp >= ?
            GROUP BY source_ip 
            HAVING port_count > 3
        """, (datetime.now() - timedelta(minutes=5),))
        
        return [{
            "type": "port_scanning",
            "source_ip": row["source_ip"],
            "ports": row["port_count"],
            "reason": f"端口扫描行为检测：尝试连接{row['port_count']}个不同端口"
        } for row in cursor.fetchall()]
    
    def detect_protocol_anomaly(self):
        """检测非常用SSH端口访问（非22端口）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, port, COUNT(*) as attempts
            FROM ssh_logs 
            WHERE event_type = 'connection_attempt'
              AND port != 22
              AND timestamp >= ?
            GROUP BY source_ip, port
            HAVING attempts > 2
        """, (datetime.now() - timedelta(hours=1),))
        
        return [{
            "type": "protocol_anomaly",
            "source_ip": row["source_ip"],
            "port": row["port"],
            "attempts": row["attempts"],
            "reason": f"非常用SSH端口访问：端口{row['port']}尝试{row['attempts']}次"
        } for row in cursor.fetchall()]


class HDFSAnomalyDetector:
    def __init__(self):
        self.conn = get_db()  # 假设已存在获取数据库连接的方法
        self.time_window_brute = 5  # 暴力操作检测窗口（分钟）
        self.time_window_enum = 60  # 用户枚举检测窗口（分钟）
        self.replication_threshold = 10  # 复制请求阈值（次/5分钟）
        self.deletion_threshold = 5  # 删除操作阈值（次/5分钟）
 
    def detect_service_brute_force(self):
        """
        检测组件级暴力操作：同一组件5分钟内异常服务超过3次
        （对应E3模板）
        """
        cursor = self.conn.cursor()
        query = """
            SELECT component,pid, COUNT(*) as attempt_count 
            FROM hdfs_logs 
            WHERE Eventid = 'E3' 
              AND timestamp >= ?
            GROUP BY component
            HAVING attempt_count > 3
        """
        cursor.execute(query, (datetime.now() - timedelta(minutes=self.time_window_brute),))
        
        return [{
            "type": "service_brute_force",
            "pid": row["pid"],
            "reason": f"组件级暴力操作：组件{row['component']}异常{row['attempt_count']}次"
        } for row in cursor.fetchall()]

 
    def detect_abnormal_deletions(self):
        """
        检测异常删除行为：同一进程5分钟内删除操作超过5次
        （对应E4/E8模板）
        """
        cursor = self.conn.cursor()
        query = """
            SELECT pid, COUNT(*) as del_count 
            FROM hdfs_logs 
            WHERE Eventid IN ('E4', 'E8') 
              AND timestamp >= ?
            GROUP BY pid
            HAVING del_count > ?
        """
        cursor.execute(query, (
            datetime.now() - timedelta(minutes=self.time_window_brute),
            self.deletion_threshold
        ))
        
        return [{
            "type": "abnormal_deletions",
            "pid": row["pid"],
            "reason": f"异常删除行为：进程{row['pid']}5分钟内删除{row['del_count']}次"
        } for row in cursor.fetchall()]
    
    def detect_transmission_failures(self):
        """
        检测传输失败风暴：块5分钟内传输失败超过5次
        （对应E10模板）
        """
        cursor = self.conn.cursor()
        query = """
            SELECT pid, COUNT(*) as fail_count 
            FROM hdfs_logs 
            WHERE Eventid = 'E10' 
              AND timestamp >= ?
            HAVING fail_count > 5
        """
        cursor.execute(query, (datetime.now() - timedelta(minutes=self.time_window_brute),))
        
        return [{
            "type": "transmission_failures",
            "pid": row["pid"],
            "reason": f"传输失败：5分钟内传输失败{row['fail_count']}次"
        } for row in cursor.fetchall()]

class LINUXAIDetector:
    def __init__(self):
        self.conn = get_db()

 
class LINUXRuleDetector:
    def __init__(self):
        self.conn = get_db()