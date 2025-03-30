from flask import Blueprint, request, jsonify
from app.services.log_parser import SSHLogParser
from app.models.db import get_db

log_bp = Blueprint("logs", __name__)

@log_bp.route("/upload", methods=["POST"])
def upload_log():
    data = request.json
    log_line = data.get("log_content")
    
    parsed_log = SSHLogParser.parse(log_line)
    if not parsed_log:
        return jsonify({"status": "error", "message": "Invalid SSH log format"}), 400
    
    # 存储到数据库
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ssh_logs (
            timestamp, process_id, event_type, 
            user_name, source_ip, port, protocol
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        parsed_log["timestamp"],
        parsed_log["process_id"],
        parsed_log["event_type"],
        parsed_log.get("user"),
        parsed_log.get("source_ip"),
        parsed_log.get("port"),
        parsed_log.get("protocol")
    ))
    conn.commit()
    
    return jsonify({"status": "success"})