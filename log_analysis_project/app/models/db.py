#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库连接模块
提供数据库连接和初始化功能
"""

import sqlite3

DB_NAME = "logs.db"

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表"""
    conn = get_db()
    cursor = conn.cursor()

    # 创建日志表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        source TEXT,
        raw_text TEXT,
        event_type TEXT,
        user TEXT,
        status TEXT,
        src_ip TEXT,
        dst_ip TEXT,
        port INTEGER,
        is_anomaly BOOLEAN
    )""")

    # 创建异常检测表
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS anomalies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_id INTEGER,
        timestamp TEXT,
        user TEXT,
        event_type TEXT,
        src_ip TEXT,
        reason TEXT,
        FOREIGN KEY (log_id) REFERENCES logs (id)
    )""")

    conn.commit()
    conn.close() 