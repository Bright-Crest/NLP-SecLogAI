from flask import Blueprint, request, jsonify
from app.services.log_parser import SSHLogParser,WebLogParser,IptablesLogParser,MySQLLogParser,HDFSLogParser,LINUXLogParser
from app.models.db import get_db
import sqlite3

log_bp = Blueprint("logs", __name__)

@log_bp.route("/upload", methods=["POST"])
def upload_log():
    data = request.json
    log_line = data.get("log_content")
    log_type = data.get("log_type")  # 区分日志类型

    if log_type == "ssh":
        parsed_log = SSHLogParser.parse(log_line)
        if not parsed_log:
            return jsonify({"status": "error", "message": "Invalid SSH log format"}), 400
    elif log_type == "web":
        parsed_log = WebLogParser.parse(log_line)
        if not parsed_log:
            return jsonify({"status": "error", "message": "Invalid web log format"}), 400
    elif log_type == "firewall":
        parsed_log = IptablesLogParser.parse(log_line)
        if not parsed_log:
            return jsonify({"status": "error", "message": "Invalid iptables log format"}), 400
    elif log_type == "mysql":
        parsed_log = MySQLLogParser.parse(log_line)
        if not parsed_log:
            return jsonify({"status": "error", "message": "Invalid MYSQL log format"}), 400
    elif log_type == "hdfs":
        parsed_log = HDFSLogParser().parse(log_line)
        if not parsed_log:
            return jsonify({"status": "error", "message": "Invalid HDFS log format"}), 400
    elif log_type == "linux":
        parsed_log = LINUXLogParser().parse(log_line)
        if not parsed_log:
            return jsonify({"status": "error", "message": "Invalid LINUX log format"}), 400
    else:
        return jsonify({"status": "error", "message": "Unsupported log type"}), 400
    
    
    # 存储到数据库
    conn = get_db()
    cursor = conn.cursor()
    # cursor.execute(""" DELETE FROM ssh_logs""")
    # cursor.execute(""" DELETE FROM ssh_anomalies""")
    if log_type == "ssh":
        cursor.execute("""
            INSERT INTO ssh_logs (
                timestamp, process_id, event_type, 
                user, source_ip, port,forwarded_ports,country_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            parsed_log["timestamp"],
            parsed_log["process_id"],
            parsed_log["event_type"],
            parsed_log.get("user"),
            parsed_log.get("source_ip"),
            parsed_log.get("port"),
            parsed_log.get("forwarded_ports"),
            parsed_log.get("country_code")
        ))
    elif log_type == "web":
        cursor.execute("""
            INSERT INTO web_logs
            (timestamp,source_ip, method, path,status_code,user_agent)
            VALUES (?, ?, ?,?,?,?)
        """, (
            parsed_log["timestamp"],
            parsed_log["source_ip"],
            parsed_log["method"],
            parsed_log["path"],
            parsed_log["status_code"],
            parsed_log["user_agent"]
        ))
    elif log_type == "firewall":
        cursor.execute("""
            INSERT INTO firewall_logs
            (timestamp,source_ip, dest_ip, protocol,src_port,dest_port,action)
            VALUES (?, ?, ?,?,?,?,?)
        """, (
            parsed_log["timestamp"],
            parsed_log["source_ip"],
            parsed_log["dest_ip"],
            parsed_log["protocol"],
            parsed_log["src_port"],
            parsed_log["dest_port"],
            parsed_log["action"]
        ))       
    elif log_type == "mysql":
        cursor.execute("""
            INSERT INTO mysql_logs
            (timestamp,user, source_ip, event_type,sql_statement,duration)
            VALUES (?, ?, ?,?,?,?)
        """, (
            parsed_log["timestamp"],
            parsed_log["user"],
            parsed_log["source_ip"],
            parsed_log["event_type"],
            parsed_log["sql_statement"],
            parsed_log["duration"]
        )) 
    elif log_type == "hdfs":
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
    elif log_type == "linux":
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

# from ..models.db import get_db_connection


@log_bp.route("/upload2", methods=["POST"])
def upload_log2():
    """上传日志（支持detected_by字段）"""
    data = request.get_json()
    required_fields = ["source_ip", "event_type", "message"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        with get_db() as conn:
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

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(base_query, params)
        results = [dict(row) for row in cursor.fetchall()]

    return jsonify({"logs": results})

