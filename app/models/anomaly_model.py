from models.db import get_db_connection
from typing import Optional

class AnomalyModel:
    @staticmethod
    def create_anomaly(
        log_id: int,
        detected_by: str,
        rule_id: Optional[int] = None,
        ai_model_version: Optional[str] = None,
        score: Optional[float] = None,
        anomaly_type: Optional[str] = None,
        description: str = ""
    ) -> int:
        """记录异常检测结果（供规则/AI模块调用）"""
        if detected_by not in {"rules", "AI", "both"}:
            raise ValueError("detected_by must be 'rules', 'AI', or 'both'")

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO anomalies (
                    log_id, rule_id, detected_by,
                    ai_model_version, score, anomaly_type, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, rule_id, detected_by, ai_model_version, score, anomaly_type, description)
            )
            anomaly_id = cursor.lastrowid
            conn.commit()
        return anomaly_id

    @staticmethod
    def get_anomalies_by_log(log_id: int) -> list:
        """获取某条日志关联的所有异常"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.*, r.name as rule_name
                FROM anomalies a
                LEFT JOIN rules r ON a.rule_id = r.rule_id
                WHERE a.log_id = ?
                ORDER BY a.created_at DESC
                """,
                (log_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def get_recent_high_risk_anomalies(score_threshold: float = 0.7, limit: int = 100) -> list:
        """获取高风险异常（供告警系统调用）"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.*, l.source_ip, l.event_type
                FROM anomalies a
                JOIN logs l ON a.log_id = l.log_id
                WHERE a.score >= ? OR a.detected_by = 'rules'
                ORDER BY a.score DESC
                LIMIT ?
                """,
                (score_threshold, limit)
            )
            return [dict(row) for row in cursor.fetchall()]