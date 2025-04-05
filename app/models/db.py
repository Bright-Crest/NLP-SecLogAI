import sqlite3

DB_NAME = "logs.db"

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
    cursor.execute("""DROP TABLE IF EXISTS ssh_logs""")
    cursor.execute("""DROP TABLE IF EXISTS ssh_anomalies""")


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

    conn.commit()
    conn.close()