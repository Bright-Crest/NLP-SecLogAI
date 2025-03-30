import re
import csv
import json
from datetime import datetime


class LogParser:
    @staticmethod
    def parse_csv(file_content: str) -> list[dict]:
        """解析CSV格式日志文件"""
        logs = []
        reader = csv.DictReader(file_content.splitlines())
        for row in reader:
            logs.append({
                "timestamp": row.get("timestamp", ""),
                "source_ip": row.get("ip") or row.get("source_ip", ""),
                "event_type": row.get("event") or row.get("event_type", ""),
                "message": row.get("message", ""),
                "raw_log": str(row)
            })
        return logs

    @staticmethod
    def parse_txt(file_content: str) -> list[dict]:
        """解析TXT格式日志文件（正则匹配）"""
        logs = []
        # 示例正则：匹配 "2023-01-01 12:00:00 [WARN] 192.168.1.1 - Message"
        pattern = r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<event_type>\w+)\] (?P<source_ip>\d+\.\d+\.\d+\.\d+) - (?P<message>.+)"

        for line in file_content.splitlines():
            match = re.match(pattern, line)
            if match:
                logs.append({
                    **match.groupdict(),
                    "raw_log": line
                })
        return logs

    @staticmethod
    def parse_jsonl(file_content: str) -> list[dict]:
        """解析JSON Lines格式日志"""
        logs = []
        for line in file_content.splitlines():
            try:
                log = json.loads(line)
                logs.append({
                    "timestamp": log.get("time") or log.get("timestamp", ""),
                    "source_ip": log.get("ip") or log.get("source_ip", ""),
                    "event_type": log.get("event") or log.get("event_type", ""),
                    "message": log.get("message", ""),
                    "raw_log": line
                })
            except json.JSONDecodeError:
                continue
        return logs