from flask import Blueprint, request, jsonify
from app.services.log_parser import SSHLogParser,WebLogParser,IptablesLogParser,MySQLLogParser
from app.models.db import get_db

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
    conn.commit()
    
    return jsonify({"status": "success"})