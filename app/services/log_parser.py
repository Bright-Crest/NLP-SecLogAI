import re
from datetime import datetime
from geoip2.database import Reader

geoip_reader = Reader('geolite/GeoLite2-City.mmdb')

class SSHLogParser:
    """
    SSH日志解析器（支持OpenSSH格式）
    示例日志：
    Dec 10 09:32:20 server sshd[1234]: Failed password for invalid user admin from 192.168.1.100 port 22 ssh2
    """
    
    @staticmethod
    def parse(log_line):
        pattern = (
            r"(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})"  # 时间戳
            r".*sshd\[(?P<process_id>\d+)\]:\s+"              # 进程ID
            r"(?P<message>.*)"                                # 日志内容
        )
        match = re.match(pattern, log_line)
        if not match:
            return None
            
        log_data = match.groupdict()
        
        # 解析年份（SSH日志不包含年份，这里手动填充当前年份）
        current_year = datetime.now().year
        log_data["timestamp"] = datetime.strptime(
            f"{current_year} {log_data['timestamp']}", "%Y %b %d %H:%M:%S"
        )
        
        # 解析日志内容
        content_parser = SSHLogParser._parse_message(log_data["message"])
        return {**log_data, **content_parser}

    @staticmethod
    def _parse_message(message):
        """解析具体的日志内容"""
        # 认证失败
        if "Failed password" in message:
            return {
                "event_type": "authentication_failure",
                "user": re.search(r"for( invalid user)? (\w+)", message).group(2),
                "source_ip": re.search(r"from (\d+\.\d+\.\d+\.\d+)", message).group(1),
                "port": re.search(r"port (\d+)", message).group(1)
            }
        
        # 非法用户尝试
        if "Invalid user" in message:
            return {
                "event_type": "invalid_user",
                "user": re.search(r"Invalid user (\w+)", message).group(1),
                "source_ip": re.search(r"from (\d+\.\d+\.\d+\.\d+)", message).group(1)
            }
        
        # 连接关闭
        if "Connection closed" in message:
            return {
                "event_type": "connection_closed",
                "source_ip": re.search(r"by (\d+\.\d+\.\d+\.\d+)", message).group(1)
            }

        # 反向DNS解析攻击检测
        if "reverse mapping checking getaddrinfo" in message and "POSSIBLE BREAK-IN ATTEMPT" in message:
            ip_match = re.search(r'\[(\d+\.\d+\.\d+\.\d+)\]', message)
            if ip_match:
                return {
                    "event_type": "reverse_dns_failure",
                    "source_ip": ip_match.group(1),
                    "reason": "反向DNS解析失败，存在潜在攻击"
                }

        # 成功登录事件
        if "Accepted password" in message:
            ip_match = re.search(r"from (\d+\.\d+\.\d+\.\d+)", message).group(1)
            return {
                "event_type": "login_success",
                "user": re.search(r"for (\w+)", message).group(1),
                "source_ip": re.search(r"from (\d+\.\d+\.\d+\.\d+)", message).group(1),
                "port": re.search(r"port (\d+)", message).group(1),
                "country_code":SSHLogParser._get_country_code(ip_match)
            }

        # 端口转发检测，目前没找到实例
        if "forwarding" in message:
            port_match = re.search(r"port (\d+)->(.+):(\d+)", message)
            if port_match:
                return {
                    "event_type": "port_forward",
                    "forwarded_ports": f"{port_match.group(1)}->{port_match.group(2)}:{port_match.group(3)}"
                }
    

        return {"event_type": "other", "raw_message": message}
    
    @staticmethod
    def _get_country_code(ip):
        """获取国家代码"""
        try:
            return geoip_reader.city(ip).country.iso_code
        except:
            return "UNKNOWN"
        

class WebLogParser:
    """
    Apache/Nginx日志解析器
    示例日志：
    192.168.1.100 - - [24/Mar/2025:12:34:56 +0800] "GET /search?q=1' OR 1=1 HTTP/1.1" 404 1234
    """
    
    @staticmethod
    def parse(log_line):
        # 匹配Apache组合日志格式（Combined Log Format）
        pattern = (
            r'(?P<source_ip>\S+) \S+ \S+ \[(?P<timestamp>.*?)\] '
            r'"(?P<method>\S+) (?P<path>.*?) HTTP/\d\.\d" '
            r'(?P<status_code>\d{3}) (?P<bytes_sent>\d+) '
            r'"(?P<referrer>.*?)" "(?P<user_agent>.*?)"'
        )
        match = re.match(pattern, log_line)
        if not match:
            return None
            
        log_data = match.groupdict()
        
        # 转换时间戳格式
        log_data["timestamp"] = datetime.strptime(
            log_data["timestamp"], "%d/%b/%Y:%H:%M:%S %z"
        )
        
        return log_data


class IptablesLogParser:
    """
    iptables日志解析器
    示例日志：
    Mar 24 12:34:56 kernel: [UFW BLOCK] IN=eth0 OUT= MAC=... SRC=192.168.1.100 DST=10.0.0.1 LEN=40 PROTO=TCP SPT=1234 DPT=80 SYN
    """
    
    @staticmethod
    def parse(log_line):
        pattern = (
            r"\w{3} \d{1,2} \d{2}:\d{2}:\d{2} .*? "
            r"\[UFW (?P<action>\w+)\] "  
            r"IN=(?P<in_interface>\S+)? "
            r"OUT=(?P<out_interface>\S+)? "
            r"SRC=(?P<source_ip>\S+) "
            r"DST=(?P<dest_ip>\S+) "
            r".*?PROTO=(?P<protocol>\S+) "
            r"SPT=(?P<src_port>\d+) "
            r"DPT=(?P<dest_port>\d+)"
        )
        match = re.search(pattern, log_line)
        if not match:
            return None
        
        log_data = match.groupdict()
        log_data["timestamp"] = datetime.now()  # 实际应为日志提取时间
        
        # 转换端口为整数
        log_data["src_port"] = int(log_data["src_port"])
        log_data["dest_port"] = int(log_data["dest_port"])
        
        return log_data
    

class MySQLLogParser:
    """
    MySQL日志解析器（支持错误日志和通用查询日志）
    示例日志：
    2025-03-24T12:34:56.123456Z 3 [Note] Access denied for user 'root'@'192.168.1.100' (using password: YES)
    2025-03-24T12:34:57.789012Z 5 [Query] SELECT * FROM customers WHERE email='test@example.com'
    """
    
    @staticmethod
    def parse(log_line):
        # 匹配通用日志格式
        pattern = (
            r"(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+"  # 时间戳
            r"\d+\s+\[(?P<event_type>\w+)\]\s+"  # 事件类型（Note/Query/Warning）
            r"(?P<message>.*)"  # 日志内容
        )
        match = re.match(pattern, log_line)
        if not match:
            return None
        
        log_data = match.groupdict()
        log_data["timestamp"] = datetime.strptime(
            log_data["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        
        # 解析消息详情（补充缺失字段）
        log_data.update(MySQLLogParser._parse_message(log_data["message"]))
        return log_data

    @staticmethod
    def _parse_message(message):
        details = {
            "user": None,
            "source_ip": None,
            "sql_statement": None,
            "duration": None
        }
        
        # 1. 登录失败事件
        if "Access denied" in message:
            user_match = re.search(r"user '(?P<user>\w+)'@'(?P<source_ip>\S+)'", message)
            if user_match:
                details.update(user_match.groupdict())
        
        # 2. SQL查询语句（包含高危操作）
        elif "Query" in message:
            sql_match = re.search(r"Query:\s+(?P<sql>.*)", message)
            if sql_match:
                details["sql_statement"] = sql_match.group("sql")
        
        # 3. 慢查询日志
        elif "Query took" in message:
            time_match = re.search(r"Query took (?P<duration>\d+\.\d+) sec", message)
            if time_match:
                details["duration"] = float(time_match.group("duration"))
        
        return details