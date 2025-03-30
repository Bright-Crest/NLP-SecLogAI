import re
from datetime import datetime

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
        
        # 解析年份（SSH日志不包含年份，需要手动补充）
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
                "port": re.search(r"port (\d+)", message).group(1),
                "protocol": re.search(r"(ssh\d)", message).group(1)
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
            
        return {"event_type": "other", "raw_message": message}