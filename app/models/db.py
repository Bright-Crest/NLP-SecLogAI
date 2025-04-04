import sqlite3

DB_NAME = "logs.db"

def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()

 
    # 创建SSH专用日志表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ssh_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME NOT NULL,
        process_id INTEGER,
        event_type TEXT NOT NULL,
        user_name TEXT,
        source_ip TEXT,
        port INTEGER,
        protocol TEXT
    )""")


    # 创建SSH异常表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS ssh_anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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
        anomaly_type TEXT NOT NULL,    -- 异常类型（如：权限异常、频繁错误等）
        details TEXT         -- 异常详情（包含原始日志ID和内容片段）
    )""")
 

    conn.commit()
    conn.close()