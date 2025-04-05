from flask import Blueprint, request, jsonify
from app.services.log_parser import SSHLogParser,HDFSLogParser
from app.models.db import get_db

log_bp = Blueprint("logs", __name__)

@log_bp.route("/upload/ssh", methods=["POST"])
def upload_ssh_log():
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
@log_bp.route("/upload/hdfs", methods=["POST"])
def upload_hdfs_log():
    data = request.json
    log_line = data.get("log_content")
    
    parsed_log = HDFSLogParser.parse(log_line)  # 假设已实现的解析类
    if not parsed_log:
        return jsonify({"status": "error", "message": "Invalid HDFS log format"}), 400
    
    # 存储到数据库（假设表结构不同）
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO hdfs_logs (
            timestamp, pid, level, 
            component, content, Eventid
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        parsed_log["timestamp"],
        parsed_log["pid"],
        parsed_log.get("level"),
        parsed_log.get("component"),
        parsed_log.get("content"),
        parsed_log.get("type")
    ))
    conn.commit()
    
    return jsonify({"status": "success"})

@log_bp.route("/upload/linux", methods=["POST"])
def upload_ssh_log():
    data = request.json
    log_line = data.get("log_content")
    
    parsed_log = SSHLogParser.parse(log_line)
    if not parsed_log:
        return jsonify({"status": "error", "message": "Invalid LINUX log format"}), 400
    
    # 存储到数据库
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO linux_logs (
            month,date,time,level,          
            component,pid,content 
        # ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        parsed_log.get("month"),
        parsed_log.get("date"),
        parsed_log.get("time"),
        parsed_log.get("level"),
        parsed_log.get("component"),
        parsed_log.get("pid"),
        parsed_log.get("content")
    ))
    conn.commit()
    
    return jsonify({"status": "success"})