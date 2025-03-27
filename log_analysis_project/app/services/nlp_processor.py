#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP处理器模块
将自然语言查询转换为SQL查询
使用OpenRouter API进行NLP处理
"""

import os
import re
import json
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 获取OpenRouter API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class NL2SQLConverter:
    """自然语言转SQL转换器"""
    
    def __init__(self, 
                 model_name: str = "anthropic/claude-3-opus", 
                 table_schema: Optional[Dict[str, Any]] = None,
                 max_tokens: int = 500,
                 temperature: float = 0.1):
        """
        初始化转换器
        
        Args:
            model_name: OpenRouter模型名称
            table_schema: 数据库表结构
            max_tokens: 最大生成令牌数
            temperature: 生成温度
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 设置默认的表结构
        if table_schema is None:
            self.table_schema = {
                "logs": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "日志ID"},
                        {"name": "timestamp", "type": "TIMESTAMP", "description": "日志时间戳"},
                        {"name": "source", "type": "TEXT", "description": "日志来源（如windows、firewall）"},
                        {"name": "raw_text", "type": "TEXT", "description": "原始日志文本"},
                        {"name": "event_type", "type": "TEXT", "description": "事件类型（如logon、connection_blocked）"},
                        {"name": "user", "type": "TEXT", "description": "用户名"},
                        {"name": "status", "type": "TEXT", "description": "状态（success或failure）"},
                        {"name": "src_ip", "type": "TEXT", "description": "源IP地址"},
                        {"name": "dst_ip", "type": "TEXT", "description": "目标IP地址"},
                        {"name": "port", "type": "INTEGER", "description": "端口号"},
                        {"name": "is_anomaly", "type": "BOOLEAN", "description": "是否为异常"}
                    ]
                },
                "anomalies": {
                    "columns": [
                        {"name": "id", "type": "INTEGER", "description": "异常ID"},
                        {"name": "log_id", "type": "INTEGER", "description": "关联的日志ID"},
                        {"name": "timestamp", "type": "TIMESTAMP", "description": "异常发生时间"},
                        {"name": "user", "type": "TEXT", "description": "相关用户"},
                        {"name": "event_type", "type": "TEXT", "description": "事件类型"},
                        {"name": "src_ip", "type": "TEXT", "description": "源IP地址"},
                        {"name": "reason", "type": "TEXT", "description": "异常原因"}
                    ]
                }
            }
        else:
            self.table_schema = table_schema
    
    def convert(self, query_text: str) -> Dict[str, Any]:
        """
        将自然语言查询转换为SQL
        
        Args:
            query_text: 自然语言查询文本
            
        Returns:
            包含SQL查询和置信度的字典
        """
        # 预处理查询文本
        processed_query = self._preprocess_query(query_text)
        
        # 构建提示
        prompt = self._build_prompt(processed_query)
        
        try:
            # 调用模型
            if OPENROUTER_API_KEY:
                result = self._call_openrouter_api(prompt)
            else:
                # 如果没有API密钥，使用模式匹配生成简单SQL
                result = self._pattern_matching_fallback(processed_query)
            
            # 提取并验证SQL
            sql = self._extract_sql(result)
            
            # 返回结果
            return {
                "sql": sql,
                "confidence": 0.9,  # 固定置信度或使用模型返回的置信度
                "original_query": query_text
            }
        except Exception as e:
            # 出错时，使用模式匹配回退方法
            fallback_result = self._pattern_matching_fallback(processed_query)
            return {
                "sql": fallback_result,
                "confidence": 0.5,  # 回退方法的置信度较低
                "original_query": query_text,
                "error": str(e)
            }
    
    def _preprocess_query(self, query_text: str) -> str:
        """预处理查询文本"""
        # 将时间表达式规范化
        query_text = re.sub(r'过去(\d+)小时', r'在最近\1小时', query_text)
        query_text = re.sub(r'最近(\d+)天', r'在最近\1天', query_text)
        query_text = re.sub(r'最近(\d+)分钟', r'在最近\1分钟', query_text)
        
        # 处理常见别名
        query_text = query_text.replace('登录失败', 'status=failure AND event_type=logon')
        
        return query_text
    
    def _build_prompt(self, query_text: str) -> str:
        """构建提示"""
        # 表结构字符串
        schema_str = ""
        for table_name, table_info in self.table_schema.items():
            schema_str += f"表名: {table_name}\n"
            schema_str += "列:\n"
            for column in table_info["columns"]:
                schema_str += f"- {column['name']} ({column['type']}): {column['description']}\n"
        
        # 构建完整提示
        prompt = f"""
你是一个专门将自然语言转换为SQL查询的AI助手。根据以下数据库表结构和查询，生成一个有效的SQL查询语句。

数据库结构:
{schema_str}

用户查询: {query_text}

请生成一个SQL查询来满足用户的要求。只返回SQL查询语句，不需要任何其他解释。确保SQL语句对SQLite是有效的。使用标准SQL时间函数如datetime('now', '-X day/hour/minute')处理时间。
        """
        
        return prompt
    
    def _call_openrouter_api(self, prompt: str) -> str:
        """调用OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://example.com", # 您的网站URL
            "X-Title": "NLP-SecLogAI"  # 您的应用名称
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个专门将自然语言转换为SQL查询的AI助手。只输出SQL查询语句。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
        response.raise_for_status()  # 如果请求失败则引发异常
        
        response_data = response.json()
        
        # 提取生成的SQL
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            raise ValueError("无法从API响应中获取内容")
    
    def _extract_sql(self, text: str) -> str:
        """从模型输出中提取SQL语句"""
        # 如果输出包含代码块，提取其中的SQL
        sql_match = re.search(r'```sql\n([\s\S]*?)\n```', text)
        if sql_match:
            return sql_match.group(1).strip()
        
        # 如果没有代码块，但有SELECT
        sql_match = re.search(r'(SELECT[\s\S]*?;)', text)
        if sql_match:
            return sql_match.group(1).strip()
        
        # 如果上述都没匹配到，返回整个文本
        return text.strip()
    
    def _pattern_matching_fallback(self, query_text: str) -> str:
        """模式匹配回退方法，当API调用失败时使用"""
        # 查找时间范围
        time_range_match = re.search(r'最近(\d+)小时', query_text)
        time_range = 24  # 默认24小时
        if time_range_match:
            time_range = int(time_range_match.group(1))
        
        # 查找用户名
        user_match = re.search(r'用户[:\s]*([^\s]+)', query_text) or re.search(r'([^\s]+)用户', query_text)
        user_condition = ""
        if user_match:
            user = user_match.group(1)
            user_condition = f"AND user = '{user}' "
        
        # 查找失败次数
        count_match = re.search(r'(失败|失效|错误)次数', query_text)
        
        # 基本查询模板
        if '登录失败' in query_text or count_match:
            return f"""
SELECT COUNT(*) as failure_count 
FROM logs 
WHERE timestamp >= datetime('now', '-{time_range} hour') 
AND status = 'failure' 
AND event_type = 'logon' 
{user_condition}
GROUP BY user 
ORDER BY failure_count DESC;
            """.strip()
        elif '攻击' in query_text or '异常' in query_text:
            return f"""
SELECT timestamp, src_ip, event_type, raw_text 
FROM logs 
WHERE timestamp >= datetime('now', '-{time_range} hour') 
AND is_anomaly = 1 
ORDER BY timestamp DESC;
            """.strip()
        else:
            # 默认查询，列出最近日志
            return f"""
SELECT timestamp, event_type, user, status, raw_text 
FROM logs 
WHERE timestamp >= datetime('now', '-{time_range} hour') 
ORDER BY timestamp DESC 
LIMIT 100;
            """.strip()


def convert_to_sql(query_text: str) -> str:
    """
    将自然语言查询转换为SQL查询
    
    Args:
        query_text: 自然语言查询文本
        
    Returns:
        SQL查询字符串
    """
    converter = NL2SQLConverter()
    result = converter.convert(query_text)
    return result["sql"] 