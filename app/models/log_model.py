from ..models.db import get_db_connection  # 使用相对导入
import re
from datetime import datetime


class LogModel:
    @staticmethod
    def create_log(source_ip: str, event_type: str, message: str, detected_by: str = "none",timestamp: datetime = None) -> int:
        """插入一条日志（自动校验IP格式）"""
        if not re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", source_ip):
            raise ValueError("Invalid IP address format")

        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S") if timestamp else None

        with get_db_connection() as conn:
            cursor = conn.cursor()
            if timestamp_str:
                cursor.execute(
                    """
                    INSERT INTO logs (source_ip, event_type, message, detected_by, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (source_ip, event_type, message, detected_by, timestamp_str)
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO logs (source_ip, event_type, message, detected_by)
                    VALUES (?, ?, ?, ?)
                    """,
                    (source_ip, event_type, message, detected_by)
                )
            log_id = cursor.lastrowid
            conn.commit()
        return log_id

    @staticmethod
    def update_detection_status(log_id: int, detected_by: str):
        """更新日志的检测状态（供规则/AI模块调用）"""
        valid_statuses = {"none", "rules", "AI", "both"}
        if detected_by not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of {valid_statuses}")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE logs SET detected_by = ? WHERE log_id = ?",
                (detected_by, log_id)
            )
            conn.commit()

    @staticmethod
    def get_logs_by_filters(
            source_ip: str = None,
            event_type: str = None,
            detected_by: str = None,
            start_time: datetime = None,
            end_time: datetime = None
    ) -> list:
        """多条件查询日志（支持分页/时间范围）"""
        query = "SELECT * FROM logs WHERE 1=1"
        params = []

        if source_ip:
            query += " AND source_ip = ?"
            params.append(source_ip)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if detected_by:
            query += " AND detected_by = ?"
            params.append(detected_by)
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.strftime("%Y-%m-%d %H:%M:%S"))  # 统一时间格式
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.strftime("%Y-%m-%d %H:%M:%S"))  # 统一时间格式

        query += " ORDER BY timestamp DESC"

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]