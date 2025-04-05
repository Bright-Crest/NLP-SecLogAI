import re
import csv
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
class HDFSLogParser:
    """
    HDFS日志解析器（支持常见HDFS日志格式）
    示例日志：
    81109\t203518\t143\tINFO\tdfs.DataNode$DataXceiver\tReceiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
    """
    def __init__(self):
        # 定义所有日志类型的正则表达式模式（保持不变）
        self.patterns = {
            'E1': r'Served block (\S+) to (\S+)',
            'E2': r'Starting thread to transfer block (\S+) to (\S+)',
            'E3': r'(\S+):Got exception while serving (\S+) to (\S+):',
            'E4': r'BLOCK\* ask (\S+) to delete (\S+)',
            'E5': r'BLOCK\* ask (\S+) to replicate (\S+) to datanode\(s\) (\S+)',
            'E6': r'BLOCK\* NameSystem\.addStoredBlock: blockMap updated: (\S+) is added to (\S+) size (\S+)',
            'E7': r'BLOCK\* NameSystem\.allocateBlock: (\S+) (\S+)',
            'E8': r'BLOCK\* NameSystem\.delete: (\S+) is added to invalidSet of (\S+)',
            'E9': r'Deleting block (\S+) file (\S+)',
            'E10': r'PacketResponder (\S+) for block (\S+) terminating',
            'E11': r'Received block (\S+) of size (\S+) from (\S+)',
            'E12': r'Received block (\S+) src: (\S+) dest: (\S+) of size (\S+)',
            'E13': r'Receiving block (\S+) src: (\S+) dest: (\S+)',
            'E14': r'Verification succeeded for (\S+)'
        }
 
    @staticmethod
    def parse(log_line):
        # 正则表达式模式（日期解析部分修改）
        pattern = re.compile(
            r"^"
            r"(?P<date>\d{5})\t"          # 日期字段保持5位（YMMDD格式）
            r"(?P<time>\d{6})\t"          # 时间字段保持6位
            r"(?P<pid>\d+)\t"
            r"(?P<level>\w+)\t"
            r"(?P<component>.*?)\t"
            r"(?P<message>.*)$"
        )
 
        match = pattern.match(log_line)
        if not match:
            return None
        log_data = match.groupdict()
 
        # 解析日期（修改部分）
        date_str = log_data["date"]
        year = date_str[0]                # 取第1位作为年份（20XX）
        month = date_str[1:3]             # 取2-3位作为月份
        day = date_str[3:5]               # 取4-5位作为日期
        parsed_date = f"200{year}-{month}-{day}"
        
        # 解析时间（保持不变）
        time_str = log_data["time"]
        hour = time_str[:2]
        minute = time_str[2:4]
        second = time_str[4:]
        parsed_time = f"{hour}:{minute}:{second}"
        
        # 拼接时间戳
        timestamp = f"{parsed_date} {parsed_time}"
        log_data["timestamp"] = timestamp
 
        # 解析日志内容（保持不变）
        content_parser = HDFSLogParser._parse_message(log_data["message"])
        return {**log_data, **content_parser}
 
    # _parse_message方法保持不变
    def _parse_message(self, message):
        for log_type, pattern in self.patterns.items():
            match = re.match(pattern, message)
            if match:
                return {
                    "type": log_type,
                }
        return None
class LINUXLogParser:
    """
    LINUX日志解析器（适配自定义格式）
    示例日志：
    Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=218.188.2.4 user=root
    """
    
    @staticmethod
    def parse(log_line):
        # 正则表达式模式匹配
        pattern = (
            r"(?P<month>\w{3})\s+"                     # 月份
            r"(?P<day>\d{1,2})\s+"                     # 日期
            r"(?P<time>\d{2}:\d{2}:\d{2})\s+"          # 时间
            r"(?P<level>\w+)\s+"                       # 日志级别
            r"(?P<component>sshd\(\w+\))\s+"           # 组件（包含pam模块）
            r"\[(?P<pid>\d+)\]:\s+"                    # 进程ID
            r"(?P<content>.*)"                         # 日志内容
        )
        
        match = re.match(pattern, log_line)
        if not match:
            return None
            
        log_data = match.groupdict()        
        # 解析日志内容
        content_parser = LINUXLogParser._parse_message(log_data["content"])
        return {**log_data,**content_parser}
 
    @staticmethod
    def _parse_message(message):
        """解析具体的日志内容"""
        result = {"event_type": "other", "raw_message": message}
        
        # 通用字段提取
        fields = message.split("; ")
        for field in fields:
            if "logname=" in field:
                result["logname"] = field.split("=")[1]
            if "uid=" in field:
                result["uid"] = field.split("=")[1]
            if "euid=" in field:
                result["euid"] = field.split("=")[1]
            if "tty=" in field:
                result["tty"] = field.split("=")[1]
            if "ruser=" in field:
                result["ruser"] = field.split("=")[1]
            if "rhost=" in field:
                result["source_ip"] = field.split("=")[1]
            if "user=" in field:
                result["user"] = field.split("=")[1]
 
        # 事件类型判断
        if "authentication failure" in message:
            result["event_type"] = "authentication_failure"
            result["success"] = False
        elif "check pass" in message:
            result["event_type"] = "password_check"
            result["success"] = "user unknown" not in message
        elif "Connection closed" in message:
            result["event_type"] = "connection_closed"
        
        return result
