#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试配置文件
包含测试中使用的全局固件和辅助函数
"""

import os
import sys
import pytest
from datetime import datetime

# 添加应用目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))

# 示例测试数据
@pytest.fixture
def sample_queries():
    """返回用于测试的示例查询"""
    return {
        "login_failure": "最近24小时admin登录失败次数",
        "anomaly": "查找所有可疑活动",
        "rules": "显示所有规则",
        "ip_specific": "来自192.168.1.100的所有活动",
        "time_range": "从2023-01-01到2023-01-31的日志",
        "user_specific": "用户root的所有操作",
        "today": "今天的异常活动",
        "complex": "上周被AI检测出的异常行为中涉及到admin用户的事件"
    }

@pytest.fixture
def sample_time_ranges():
    """返回用于测试的时间范围示例"""
    now = datetime.now()
    return {
        "relative": {"type": "relative", "unit": "hour", "value": 24},
        "special_today": {"type": "special", "start": datetime(now.year, now.month, now.day, 0, 0, 0), "end": datetime(now.year, now.month, now.day, 23, 59, 59)},
        "absolute": {"type": "absolute", "start": datetime(2023, 1, 1), "end": datetime(2023, 1, 31)}
    }

@pytest.fixture
def sample_db_schema():
    """返回示例数据库模式"""
    return {
        "logs": {
            "columns": [
                {"name": "log_id", "type": "INTEGER", "description": "日志唯一ID"},
                {"name": "timestamp", "type": "DATETIME", "description": "日志时间戳"},
                {"name": "source_ip", "type": "TEXT", "description": "来源IP地址"},
                {"name": "event_type", "type": "TEXT", "description": "事件类型(login, file_access等)"},
                {"name": "message", "type": "TEXT", "description": "日志原始内容"},
                {"name": "detected_by", "type": "TEXT", "description": "检测方式(rules, AI, both, none)"},
                {"name": "created_at", "type": "TIMESTAMP", "description": "记录创建时间"}
            ]
        },
        "rules": {
            "columns": [
                {"name": "rule_id", "type": "INTEGER", "description": "规则唯一ID"},
                {"name": "name", "type": "TEXT", "description": "规则名称"},
                {"name": "description", "type": "TEXT", "description": "规则描述"},
                {"name": "sql_query", "type": "TEXT", "description": "规则SQL查询语句"},
                {"name": "action", "type": "TEXT", "description": "规则动作(alert, block, log)"},
                {"name": "created_at", "type": "TIMESTAMP", "description": "规则创建时间"}
            ]
        },
        "anomalies": {
            "columns": [
                {"name": "anomaly_id", "type": "INTEGER", "description": "异常唯一ID"},
                {"name": "log_id", "type": "INTEGER", "description": "关联的日志ID"},
                {"name": "rule_id", "type": "INTEGER", "description": "关联的规则ID(若由规则检测)"},
                {"name": "detected_by", "type": "TEXT", "description": "检测方式(rules, AI, both)"},
                {"name": "ai_model_version", "type": "TEXT", "description": "AI模型版本"},
                {"name": "score", "type": "REAL", "description": "AI置信度评分"},
                {"name": "anomaly_type", "type": "TEXT", "description": "异常类型"},
                {"name": "description", "type": "TEXT", "description": "异常详细描述"},
                {"name": "created_at", "type": "TIMESTAMP", "description": "记录创建时间"}
            ]
        }
    } 