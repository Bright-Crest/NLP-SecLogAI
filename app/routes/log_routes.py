from flask import Blueprint, request, jsonify
from ..models.db import get_db_connection

log_bp = Blueprint("logs", __name__)


@log_bp.route("/upload", methods=["POST"])
def upload_log():
    """上传日志（支持detected_by字段）"""
    data = request.get_json()
    required_fields = ["source_ip", "event_type", "message"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO logs (source_ip, event_type, message, detected_by)
                VALUES (?, ?, ?, ?)
                """,
                (data["source_ip"], data["event_type"], data["message"], data.get("detected_by", "none"))
            )
            log_id = cursor.lastrowid
            conn.commit()

        return jsonify({"status": "success", "log_id": log_id})

    except sqlite3.IntegrityError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500


@log_bp.route("/query", methods=["POST"])
def query_logs():
    """查询日志（支持规则检测和AI检测标记过滤）"""
    filters = request.get_json().get("filters", {})

    base_query = "SELECT * FROM logs WHERE 1=1"
    params = []

    # 动态构建查询条件
    if "source_ip" in filters:
        base_query += " AND source_ip = ?"
        params.append(filters["source_ip"])
    if "event_type" in filters:
        base_query += " AND event_type = ?"
        params.append(filters["event_type"])
    if "detected_by" in filters:
        base_query += " AND detected_by = ?"
        params.append(filters["detected_by"])

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(base_query, params)
        results = [dict(row) for row in cursor.fetchall()]

    return jsonify({"logs": results})