from app.models.db import get_db
from datetime import datetime, timedelta
from geoip2.database import Reader
from collections import defaultdict

class SSHRuleDetector:
    def __init__(self):
        self.conn = get_db()
        self.geoip_reader = Reader('geolite/GeoLite2-City.mmdb')
    
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
            SELECT source_ip, COUNT(DISTINCT user) as user_count
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

    def detect_reverse_dns_failure(self):
        """检测反向DNS解析失败告警"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(*) as total 
            FROM ssh_logs 
            WHERE event_type = 'reverse_dns_failure'
              AND timestamp >= datetime('now', '-1 hour')
            GROUP BY source_ip
        """)
        
        return [{
            "type": "reverse_dns_failure",
            "source_ip": row["source_ip"],
            "count": row["total"],
            "reason": f"反向DNS解析失败告警: {row['source_ip']} 触发{row['total']}次"
        } for row in cursor.fetchall()]

    def detect_high_frequency(self, window=1, threshold=50):
        """高频连接检测"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT source_ip, COUNT(*) as connections 
            FROM ssh_logs 
            WHERE timestamp >= datetime('now', '-{window} minutes')
            GROUP BY source_ip 
            HAVING connections > {threshold}
        """)
        
        return self._format_result(
            "high_frequency",
            cursor.fetchall(),
            lambda r: f"高频连接: {r['source_ip']} {r['connections']}次/{window}分钟"
        )

    def detect_geo_anomaly(self, hours=1):
        """地理异常检测"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user, source_ip, timestamp 
            FROM ssh_logs 
            WHERE event_type = 'login_success'
              AND timestamp >= ?
        """, (datetime.now() - timedelta(hours=hours),))
        
        user_data = defaultdict(set)
        for row in cursor.fetchall():
            country = self.get_country_code(row["source_ip"])
            user_data[row["user"]].add(country)
        
        return [{
            "type": "geo_anomaly",
            "user": user,
            "countries": list(countries),
            "reason": f"用户 {user} {len(countries)}小时内从 {len(countries)} 个国家登录"
        } for user, countries in user_data.items() if len(countries) > 1]

    def detect_port_scan(self, threshold=100):
        """端口扫描检测"""
        return self.detect_high_frequency(threshold=threshold)

    def get_country_code(self, ip):
        """获取国家代码"""
        try:
            return self.geoip_reader.city(ip).country.iso_code
        except:
            return "UNKNOWN"

    def _format_result(self, anomaly_type, rows, reason_func):
        """统一格式化结果"""
        return [{
            "type": anomaly_type,
            "source_ip": row["source_ip"],
            "user": None,
            "count": row[list(row.keys())[1]],  # 动态获取统计字段
            "reason": reason_func(row)
        } for row in rows]
    
    def run_all_rules(self):
        """执行所有规则检测"""
        anomalies = []
        anomalies.extend(self.detect_brute_force())
        anomalies.extend(self.detect_illegal_users())
        anomalies.extend(self.detect_reverse_dns_failure())
        anomalies.extend(self.detect_geo_anomaly())
        anomalies.extend(self.detect_high_frequency())
        anomalies.extend(self.detect_port_scan())
        return anomalies

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
    
    # def detect_port_scanning(self):
    #     """检测端口扫描行为（同一IP尝试连接超过3个不同端口）"""
    #     cursor = self.conn.cursor()
    #     cursor.execute("""
    #         SELECT source_ip, COUNT(DISTINCT port) as port_count
    #         FROM ssh_logs 
    #         WHERE event_type IN ('connection_attempt', 'invalid_port')
    #           AND timestamp >= ?
    #         GROUP BY source_ip 
    #         HAVING port_count > 3
    #     """, (datetime.now() - timedelta(minutes=5),))
        
    #     return [{
    #         "type": "port_scanning",
    #         "source_ip": row["source_ip"],
    #         "ports": row["port_count"],
    #         "reason": f"端口扫描行为检测：尝试连接{row['port_count']}个不同端口"
    #     } for row in cursor.fetchall()]
    
    # def detect_protocol_anomaly(self):
    #     """检测非常用SSH端口访问（非22端口）"""
    #     cursor = self.conn.cursor()
    #     cursor.execute("""
    #         SELECT source_ip, port, COUNT(*) as attempts
    #         FROM ssh_logs 
    #         WHERE event_type = 'connection_attempt'
    #           AND port != 22
    #           AND timestamp >= ?
    #         GROUP BY source_ip, port
    #         HAVING attempts > 2
    #     """, (datetime.now() - timedelta(hours=1),))
        
    #     return [{
    #         "type": "protocol_anomaly",
    #         "source_ip": row["source_ip"],
    #         "port": row["port"],
    #         "attempts": row["attempts"],
    #         "reason": f"非常用SSH端口访问：端口{row['port']}尝试{row['attempts']}次"
    #     } for row in cursor.fetchall()]


class WebLogDetector:
    def __init__(self):
        self.conn = get_db()
    
    def detect_sqli(self):
        """检测SQL注入特征"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, path 
            FROM web_logs 
            WHERE path LIKE '%UNION%SELECT%'
               OR path LIKE '%1=1%'
               OR path LIKE '%DROP%TABLE%'
        """)
        return [{
            "type": "sql_injection",
            "source_ip": row["source_ip"],
            "path": row["path"],
            "reason": f"疑似SQL注入攻击：{row['path']}"
        } for row in cursor.fetchall()]
    
    def detect_xss(self):
        """检测XSS攻击特征"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, path 
            FROM web_logs 
            WHERE path LIKE '%<script>%'
               OR path LIKE '%alert(%'
        """)
        return [{
            "type": "xss",
            "source_ip": row["source_ip"],
            "path": row["path"],
            "reason": f"疑似XSS攻击：{row['path']}"
        } for row in cursor.fetchall()]
    
    def detect_brute_force(self):
        """检测登录暴力破解（5分钟内同一IP超过20次401/403）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(*) as attempts 
            FROM web_logs 
            WHERE status_code IN (401, 403)
              AND timestamp >= ?
            GROUP BY source_ip 
            HAVING attempts > 20
        """, (datetime.now() - timedelta(minutes=5),))
        return [{
            "type": "web_brute_force",
            "source_ip": row["source_ip"],
            "attempts": row["attempts"],
            "reason": f"Web暴力破解检测：{row['attempts']}次失败尝试"
        } for row in cursor.fetchall()]
    
    def detect_scanner(self):
        """检测扫描行为（高频404请求）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(*) as errors 
            FROM web_logs 
            WHERE status_code = 404
              AND timestamp >= ?
            GROUP BY source_ip 
            HAVING errors > 50
        """, (datetime.now() - timedelta(minutes=10),))
        return [{
            "type": "scanner",
            "source_ip": row["source_ip"],
            "errors": row["errors"],
            "reason": f"疑似扫描行为：{row['errors']}次404错误"
        } for row in cursor.fetchall()]
    
    def detect_directory_traversal(self):
        """检测目录遍历攻击"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, path 
            FROM web_logs 
            WHERE path LIKE '%/../%' 
               OR path LIKE '%/etc/passwd%'
               OR path LIKE '%/win.ini%'
        """)
        return [{
            "type": "directory_traversal",
            "source_ip": row["source_ip"],
            "path": row["path"],
            "reason": f"疑似目录遍历攻击：{row['path']}"
        } for row in cursor.fetchall()]
    
    def detect_sensitive_files(self):
        """检测敏感文件访问"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, path 
            FROM web_logs 
            WHERE path LIKE '%/.git/config%'
               OR path LIKE '%/wp-admin/%'
               OR path LIKE '%/phpmyadmin/%'
        """)
        return [{
            "type": "sensitive_file_access",
            "source_ip": row["source_ip"],
            "path": row["path"],
            "reason": f"疑似敏感文件访问：{row['path']}"
        } for row in cursor.fetchall()]
    
    def detect_suspicious_ua(self):
        """检测可疑User-Agent"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, user_agent 
            FROM web_logs 
            WHERE LOWER(user_agent) LIKE '%sqlmap%'
               OR LOWER(user_agent) LIKE '%nmap%'
               OR LOWER(user_agent) LIKE '%hydra%'
        """)
        return [{
            "type": "suspicious_user_agent",
            "source_ip": row["source_ip"],
            "user_agent": row["user_agent"],
            "reason": f"可疑工具标识：{row['user_agent']}"
        } for row in cursor.fetchall()]
    
    def detect_high_frequency_requests(self):
        """检测高频请求（1分钟内同一IP请求同一端点超过100次）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, path, COUNT(*) as count 
            FROM web_logs 
            WHERE timestamp >= ?
            GROUP BY source_ip, path 
            HAVING count > 100
        """, (datetime.now() - timedelta(minutes=1),))
        return [{
            "type": "high_frequency_requests",
            "source_ip": row["source_ip"],
            "path": row["path"],
            "count": row["count"],
            "reason": f"高频请求：{row['path']} ({row['count']}次/分钟)"
        } for row in cursor.fetchall()]
    
    def detect_abnormal_methods(self):
        """检测非常用HTTP方法（如PUT/DELETE）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, method, path 
            FROM web_logs 
            WHERE method NOT IN ('GET', 'POST', 'HEAD')
        """)
        return [{
            "type": "abnormal_http_method",
            "source_ip": row["source_ip"],
            "method": row["method"],
            "path": row["path"],
            "reason": f"非常用HTTP方法：{row['method']} {row['path']}"
        } for row in cursor.fetchall()]


class FirewallDetector:
    def __init__(self):
        self.conn = get_db()
    
    def detect_port_scan(self, time_window=5, port_threshold=20):
        """检测端口扫描（同一IP在N分钟内访问超过M个不同端口）"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT source_ip, COUNT(DISTINCT dest_port) as port_count 
            FROM firewall_logs 
            WHERE timestamp >= ? AND action = 'DROP'
            GROUP BY source_ip 
            HAVING port_count > ?
        """, (datetime.now() - timedelta(minutes=time_window), port_threshold))
        return [{
            "type": "port_scan",
            "source_ip": row["source_ip"],
            "ports": row["port_count"],
            "reason": f"疑似端口扫描：{row['port_count']}个端口被探测"
        } for row in cursor.fetchall()]
    
    def detect_ddos(self, time_window=1, packet_threshold=1000):
        """检测DDoS攻击（同一IP每秒发送超过N个SYN包）"""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT source_ip, COUNT(*) as packet_count 
            FROM firewall_logs 
            WHERE protocol = 'TCP' AND dest_port = 80 
              AND timestamp >= ? AND action = 'DROP'
            GROUP BY source_ip 
            HAVING packet_count > ?
        """, (datetime.now() - timedelta(minutes=time_window), packet_threshold))
        return [{
            "type": "ddos",
            "source_ip": row["source_ip"],
            "packets": row["packet_count"],
            "reason": f"疑似DDoS攻击：{row['packet_count']}次SYN请求"
        } for row in cursor.fetchall()]
    
    # def detect_blacklist_ip(self):
    #     """检测黑名单IP访问（需预加载威胁情报数据）"""
    #     with open("config/ip_blacklist.txt") as f:
    #         blacklist = [ip.strip() for ip in f.readlines()]
        
    #     cursor = self.conn.cursor()
    #     cursor.execute("""
    #         SELECT DISTINCT source_ip 
    #         FROM firewall_logs 
    #         WHERE source_ip IN ({})
    #     """.format(",".join(["?"]*len(blacklist))), blacklist)
    #     return [{
    #         "type": "blacklist_ip",
    #         "source_ip": row["source_ip"],
    #         "reason": "命中IP黑名单"
    #     } for row in cursor.fetchall()]
    
    def detect_abnormal_protocols(self):
        """检测非常用协议（如SMB暴露在公网）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, dest_port, protocol 
            FROM firewall_logs 
            WHERE protocol = 'UDP' AND dest_port IN (137,138,139,445)
        """)
        return [{
            "type": "abnormal_protocol",
            "source_ip": row["source_ip"],
            "port": row["dest_port"],
            "reason": f"非常用协议：{row['protocol']}:{row['dest_port']}"
        } for row in cursor.fetchall()]
    
    def detect_internal_outbound(self):
        """检测内网IP异常外联（如矿池地址）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, dest_ip 
            FROM firewall_logs 
            WHERE source_ip LIKE '192.168.%' 
              AND dest_ip NOT LIKE '192.168.%'
              AND action = 'DROP'
        """)
        return [{
            "type": "internal_outbound",
            "source_ip": row["source_ip"],
            "dest_ip": row["dest_ip"],
            "reason": f"内网IP异常外联至：{row['dest_ip']}"
        } for row in cursor.fetchall()]


class MySQLDetector:
    def __init__(self):
        self.conn = get_db()
    
    def detect_brute_force(self, threshold=5):
        """检测暴力破解（同一IP多次认证失败）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT source_ip, COUNT(*) as attempts 
            FROM mysql_logs 
            WHERE event_type = 'auth_failure'
              AND timestamp >= ?
            GROUP BY source_ip 
            HAVING attempts > ?
        """, (datetime.now() - timedelta(hours=1), threshold))
        return [{
            "type": "brute_force",
            "source_ip": row["source_ip"],
            "attempts": row["attempts"],
            "reason": f"MySQL暴力破解：{row['attempts']}次失败尝试"
        } for row in cursor.fetchall()]
    
    def detect_high_risk_operations(self):
        """检测高危SQL操作"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user, source_ip, sql_statement 
            FROM mysql_logs 
            WHERE event_type = 'high_risk_operation'
              AND sql_statement LIKE '%DROP%TABLE%'
        """)
        return [{
            "type": "high_risk_sql",
            "user": row["user"],
            "source_ip": row["source_ip"],
            "sql": row["sql_statement"],
            "reason": f"高危操作：{row['sql_statement']}"
        } for row in cursor.fetchall()]
    
    def detect_privilege_escalation(self):
        """检测权限变更（如GRANT/REVOKE）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user, sql_statement 
            FROM mysql_logs 
            WHERE sql_statement LIKE '%GRANT%ALL%'
               OR sql_statement LIKE '%REVOKE%'
        """)
        return [{
            "type": "privilege_change",
            "user": row["user"],
            "sql": row["sql_statement"],
            "reason": f"权限变更操作：{row['sql_statement']}"
        } for row in cursor.fetchall()]
    
    def detect_data_exfiltration(self, row_threshold=1000):
        """检测大量数据导出"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user, source_ip, sql_statement 
            FROM mysql_logs 
            WHERE sql_statement LIKE 'SELECT%*%FROM%'
              AND sql_statement NOT LIKE '%LIMIT%'
        """)
        return [{
            "type": "data_exfiltration",
            "user": row["user"],
            "source_ip": row["source_ip"],
            "sql": row["sql_statement"],
            "reason": f"疑似数据导出：{row['sql_statement']}"
        } for row in cursor.fetchall()]
    
    def detect_off_hour_access(self):
        """检测非工作时间访问（凌晨0-5点）"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT user, source_ip, sql_statement 
            FROM mysql_logs 
            WHERE CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 0 AND 5
        """)
        return [{
            "type": "off_hour_access",
            "user": row["user"],
            "source_ip": row["source_ip"],
            "reason": f"非工作时间操作：{row['sql_statement']}"
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