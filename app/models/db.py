import sqlite3
from contextlib import contextmanager
from flask import current_app

DB_NAME = "logs.db"

TABLE_SCHEMA = {
                "ssh_logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "SSH日志唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "日志时间戳"},
                        {"name": "process_id", "type": "INTEGER", "description": "进程ID"},
                        {"name": "event_type", "type": "TEXT", "description": "事件类型"},
                        {"name": "user", "type": "TEXT", "description": "用户名"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "port", "type": "INTEGER", "description": "端口号"},
                        {"name": "country_code", "type": "TEXT", "description": "国家代码"},
                        {"name": "forwarded_ports", "type": "TEXT", "description": "端口转发信息(JSON格式)"}
                    ]
                },
                "ssh_anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "SSH异常唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "异常检测时间"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                        {"name": "details", "type": "TEXT", "description": "异常详情"},
                        {"name": "user", "type": "TEXT", "description": "关联用户"},
                        {"name": "country_codes", "type": "TEXT", "description": "国家代码"}
                    ]
                },
                "web_logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "Web日志唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "日志时间戳"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "method", "type": "TEXT", "description": "HTTP方法"},
                        {"name": "path", "type": "TEXT", "description": "请求路径"},
                        {"name": "status_code", "type": "INTEGER", "description": "HTTP状态码"},
                        {"name": "user_agent", "type": "TEXT", "description": "用户代理"}
                    ]
                },
                "web_anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "Web异常唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "异常检测时间"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                        {"name": "details", "type": "TEXT", "description": "异常详情"}
                    ]
                },
                "firewall_logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "防火墙日志唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "日志时间戳"},
                        {"name": "source_ip", "type": "TEXT", "description": "源IP地址"},
                        {"name": "dest_ip", "type": "TEXT", "description": "目标IP地址"},
                        {"name": "protocol", "type": "TEXT", "description": "协议"},
                        {"name": "src_port", "type": "INTEGER", "description": "源端口"},
                        {"name": "dest_port", "type": "INTEGER", "description": "目标端口"},
                        {"name": "action", "type": "TEXT", "description": "防火墙动作"}
                    ]
                },
                "firewall_anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "防火墙异常唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "异常检测时间"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                        {"name": "details", "type": "TEXT", "description": "异常详情"}
                    ]
                },
                "mysql_logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "MySQL日志唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "日志时间戳"},
                        {"name": "user", "type": "TEXT", "description": "数据库用户"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "event_type", "type": "TEXT", "description": "事件类型"},
                        {"name": "sql_statement", "type": "TEXT", "description": "SQL语句"},
                        {"name": "duration", "type": "FLOAT", "description": "执行时间(秒)"}
                    ]
                },
                "mysql_anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "MySQL异常唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "异常检测时间"},
                        {"name": "user", "type": "TEXT", "description": "数据库用户"},
                        {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                        {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                        {"name": "details", "type": "TEXT", "description": "异常详情"}
                    ]
                },
                "hdfs_logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "HDFS日志唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "日志时间戳"},
                        {"name": "pid", "type": "INTEGER", "description": "进程ID"},
                        {"name": "level", "type": "TEXT", "description": "日志级别"},
                        {"name": "component", "type": "TEXT", "description": "日志组件"},
                        {"name": "content", "type": "TEXT", "description": "日志内容"},
                        {"name": "Eventid", "type": "TEXT", "description": "事件ID"}
                    ]
                },
                "hdfs_anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "HDFS异常唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "异常检测时间"},
                        {"name": "pid", "type": "INTEGER", "description": "关联进程ID"},
                        {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                        {"name": "details", "type": "TEXT", "description": "异常详情"}
                    ]
                },
                "linux_logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "Linux系统日志唯一ID"},
                        {"name": "month", "type": "TEXT", "description": "月份"},
                        {"name": "date", "type": "INTEGER", "description": "日期"},
                        {"name": "time", "type": "TEXT", "description": "时间"},
                        {"name": "level", "type": "TEXT", "description": "日志级别"},
                        {"name": "component", "type": "TEXT", "description": "日志组件"},
                        {"name": "pid", "type": "INTEGER", "description": "进程ID"},
                        {"name": "content", "type": "TEXT", "description": "日志内容"}
                    ]
                },
                "linux_anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "Linux异常唯一ID"},
                        {"name": "timestamp", "type": "DATETIME", "description": "异常检测时间"},
                        {"name": "component", "type": "TEXT", "description": "日志组件"},
                        {"name": "pid", "type": "INTEGER", "description": "关联进程ID"},
                        {"name": "detected", "type": "TEXT", "description": "检测方式标记"},
                        {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                        {"name": "details", "type": "TEXT", "description": "异常详情"}
                    ]
                }
}


def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # # 创建日志表
    # cursor.execute("""
    # CREATE TABLE IF NOT EXISTS logs (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     timestamp TEXT,
    #     username TEXT,
    #     event TEXT,
    #     ip_address TEXT
    # )""")

    # # 创建异常检测表
    # cursor.execute("""
    # CREATE TABLE IF NOT EXISTS anomalies (
    #     id INTEGER PRIMARY KEY AUTOINCREMENT,
    #     timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    #     username TEXT,
    #     event TEXT,
    #     ip_address TEXT,
    #     reason TEXT,
    #     detected_by TEXT CHECK(detected_by IN ('rules', 'AI', 'both'))
    # )""")
    # cursor.execute("""DROP TABLE IF EXISTS ssh_logs""")
    # cursor.execute("""DROP TABLE IF EXISTS ssh_anomalies""")


    # 创建SSH专用日志表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ssh_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        process_id INTEGER,
        event_type TEXT NOT NULL,
        user TEXT,
        source_ip TEXT,
        port INTEGER,
        country_code TEXT,             -- 国家代码
        forwarded_ports TEXT           -- JSON存储端口转发信息
    )""")


    # 创建SSH异常表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ssh_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_ip TEXT,
        anomaly_type TEXT NOT NULL,
        details TEXT,
        user TEXT,
        country_codes TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS web_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        source_ip TEXT NOT NULL,
        method TEXT,
        path TEXT,
        status_code INTEGER,
        user_agent TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS web_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_ip TEXT,
        anomaly_type TEXT NOT NULL,
        details TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS firewall_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        source_ip TEXT NOT NULL,
        dest_ip TEXT NOT NULL,
        protocol TEXT,
        src_port INTEGER,
        dest_port INTEGER,
        action TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS firewall_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_ip TEXT,
        anomaly_type TEXT NOT NULL,
        details TEXT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mysql_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        user TEXT,
        source_ip TEXT,
        event_type TEXT NOT NULL,
        sql_statement TEXT,
        duration FLOAT
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS mysql_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user TEXT,
        source_ip TEXT,
        anomaly_type TEXT NOT NULL,
        details TEXT
    )""")

    # 创建HDFS原始日志表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hdfs_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        pid INTEGER,                   -- 进程ID
        level TEXT,           -- 日志级别
        component TEXT,       -- 日志组件
        content TEXT,          -- 日志内容
        Eventid TEXT
    )""")
 
    # 创建HDFS异常表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hdfs_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,           
        pid INTEGER,                   -- 关联进程ID
        anomaly_type TEXT NOT NULL,    -- 异常类型（如：权限异常、频繁错误等）
        details TEXT         -- 异常详情（包含原始日志ID和内容片段）
    )""")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS linux_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT,
        date INTEGER,
        time TEXT,
        level TEXT NOT NULL,           -- 日志级别           
        component TEXT NOT NULL,       -- 日志组件
        pid INTEGER,
        content TEXT NOT NULL          -- 日志内容
    )""")
 
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS linux_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        component TEXT NOT NULL,       -- 日志组件
        pid INTEGER,                   -- 关联进程ID
        detected TEXT NOT NULL,     --标记是否为AI检测
        anomaly_type TEXT ,    -- 异常类型（如：权限异常、频繁错误等）
        details TEXT         -- 异常详情（包含原始日志ID和内容片段）
    )""")
  
    # 日志表（含detected_by字段）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        source_ip TEXT NOT NULL,
        event_type TEXT NOT NULL,
        message TEXT NOT NULL,
        detected_by TEXT CHECK(detected_by IN ('none', 'rules', 'AI', 'both')) DEFAULT 'none',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # 规则表（支持action字段）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rules (
        rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        sql_query TEXT NOT NULL,
        action TEXT CHECK(action IN ('alert', 'block', 'log')) DEFAULT 'alert',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # 异常检测表（修正AUTOINCREMENT拼写错误）
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_id INTEGER NOT NULL,
        rule_id INTEGER,
        detected_by TEXT CHECK(detected_by IN ('rules', 'AI', 'both')) NOT NULL,
        ai_model_version TEXT,
        score REAL,
        anomaly_type TEXT,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (log_id) REFERENCES logs(log_id),
        FOREIGN KEY (rule_id) REFERENCES rules(rule_id)
    )""")

    # 索引优化（加速查询）
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_ip ON logs(source_ip)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomalies_log_id ON anomalies(log_id)")
  
    conn.commit()
    conn.close()




# @contextmanager
# def get_db_connection():
#     """获取数据库连接（自动管理事务和异常）"""
#     conn = sqlite3.connect(DB_NAME)
#     conn.row_factory = sqlite3.Row  # 返回字典形式的结果
#     try:
#         yield conn
#     except sqlite3.Error as e:
#         conn.rollback()
#         current_app.logger.error(f"Database error: {e}")
#         raise
#     finally:
#         conn.close()


