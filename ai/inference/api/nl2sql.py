#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自然语言转SQL模块
将用户的自然语言查询转换为结构化SQL查询
"""

import os
import re
import json
from typing import Dict, Any, List, Optional
import openai

# 加载OpenAI API密钥（从环境变量或配置文件）
openai.api_key = os.environ.get("OPENAI_API_KEY", "")


class NL2SQLConverter:
    """自然语言转SQL转换器"""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo", 
                 table_schema: Optional[Dict[str, Any]] = None,
                 max_tokens: int = 500,
                 temperature: float = 0.1):
        """
        初始化转换器
        
        Args:
            model_name: OpenAI模型名称
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
            if openai.api_key:
                result = self._call_openai_api(prompt)
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

请生成一个SQL查询来满足用户的要求。只返回SQL查询语句，不需要任何其他解释。
        """
        
        return prompt
    
    def _call_openai_api(self, prompt: str) -> str:
        """调用OpenAI API"""
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "你是一个专门将自然语言转换为SQL查询的AI助手。只输出SQL查询语句。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        # 提取生成的SQL
        return response.choices[0].message.content.strip()
    
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
        user_match = re.search(r'用户[:\s]*([^\s]+)', query_text)
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
WHERE timestamp >= NOW() - INTERVAL '{time_range} HOUR' 
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
WHERE timestamp >= NOW() - INTERVAL '{time_range} HOUR' 
AND is_anomaly = TRUE 
ORDER BY timestamp DESC;
            """.strip()
        else:
            # 默认查询，列出最近日志
            return f"""
SELECT timestamp, event_type, user, status, raw_text 
FROM logs 
WHERE timestamp >= NOW() - INTERVAL '{time_range} HOUR' 
ORDER BY timestamp DESC 
LIMIT 100;
            """.strip() 