#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP处理器测试模块
测试自然语言转SQL功能
"""

import sys
import os
import unittest

# 添加应用目录到Path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    os.path.pardir, 
    'app'
)))

from services.nlp_processor import NL2SQLConverter, convert_to_sql

class TestNL2SQLConverter(unittest.TestCase):
    """测试NL2SQL转换器"""
    
    def setUp(self):
        """初始化测试环境"""
        self.converter = NL2SQLConverter()
    
    def test_pattern_matching_fallback_login_failures(self):
        """测试模式匹配回退方法 - 登录失败"""
        query = "最近24小时admin登录失败次数"
        result = self.converter._pattern_matching_fallback(query)
        self.assertIn("SELECT COUNT(*) as failure_count", result)
        self.assertIn("FROM logs", result)
        self.assertIn("WHERE timestamp >= datetime('now', '-24 hour')", result)
        self.assertIn("AND status = 'failure'", result)
        self.assertIn("AND event_type = 'logon'", result)
        self.assertIn("AND user = 'admin'", result)
    
    def test_pattern_matching_fallback_anomalies(self):
        """测试模式匹配回退方法 - 异常事件"""
        query = "最近12小时的异常事件"
        result = self.converter._pattern_matching_fallback(query)
        self.assertIn("SELECT timestamp, src_ip, event_type, raw_text", result)
        self.assertIn("FROM logs", result)
        self.assertIn("WHERE timestamp >= datetime('now', '-12 hour')", result)
        self.assertIn("AND is_anomaly = 1", result)
    
    def test_pattern_matching_fallback_default(self):
        """测试模式匹配回退方法 - 默认查询"""
        query = "最近6小时的日志"
        result = self.converter._pattern_matching_fallback(query)
        self.assertIn("SELECT timestamp, event_type, user, status, raw_text", result)
        self.assertIn("FROM logs", result)
        self.assertIn("WHERE timestamp >= datetime('now', '-6 hour')", result)
        self.assertIn("LIMIT 100", result)
    
    def test_convert_to_sql_function(self):
        """测试convert_to_sql辅助函数"""
        # 注意：这里不实际调用API，而是使用模式匹配
        # 在实际项目中，可以使用mock来模拟API
        query = "最近24小时登录失败次数"
        sql = convert_to_sql(query)
        self.assertTrue(isinstance(sql, str))
        self.assertIn("SELECT", sql)

if __name__ == "__main__":
    unittest.main() 