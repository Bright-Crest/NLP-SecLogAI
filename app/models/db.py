import sqlite3
from contextlib import contextmanager
from flask import current_app

DB_NAME = "security_logs.db"


@contextmanager
def get_db_connection():
    """获取数据库连接（自动管理事务和异常）"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row  # 返回字典形式的结果
    try:
        yield conn
    except sqlite3.Error as e:
        conn.rollback()
        current_app.logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db():
    """初始化数据库表结构（兼容Flask-Migrate）"""
    with get_db_connection() as conn:
        cursor = conn.cursor()

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