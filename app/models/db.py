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


    conn.commit()
    conn.close()