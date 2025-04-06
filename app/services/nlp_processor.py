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
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
from datetime import datetime, timedelta

# 加载环境变量
load_dotenv()

# 获取OpenRouter API密钥
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

class NL2SQLConverter:
    """自然语言转SQL转换器"""
    
    def __init__(self, 
                 model_name: str = "openrouter/auto", # "deepseek/deepseek-r1:free" 
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
        processed_query, time_range = self._preprocess_query(query_text)
        
        # 构建提示
        prompt = self._build_prompt(processed_query, time_range)
        
        try:
            # 检查API密钥是否可用
            if not os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY").strip() == "":
                # 如果没有API密钥，直接使用模式匹配生成简单SQL
                result = self._pattern_matching_fallback(processed_query, time_range)
                
                return {
                    "sql": result,
                    "confidence": 0.7,  # 回退方法的置信度中等
                    "original_query": query_text
                }
            
            # 调用模型
            result = self._call_openrouter_api(prompt)
            
            # 提取并验证SQL
            sql = self._extract_sql(result)
            if not sql.strip():
                raise ValueError("API返回的SQL为空")
            
            # 对特定实体名称进行增强匹配
            enhanced_sql = self._enhance_entity_matching(sql, processed_query)
            
            # 返回结果
            return {
                "sql": enhanced_sql,
                "confidence": 0.9,  # 固定置信度或使用模型返回的置信度
                "original_query": query_text
            }
        except Exception as e:
            import warnings
            warnings.warn(f"AI转换SQL失败: {str(e)}。原始查询: {query_text}。使用模式匹配回退方法。")

            # 出错时，使用模式匹配回退方法
            fallback_result = self._pattern_matching_fallback(processed_query, time_range)
            
            # 确保生成结果非空
            if not fallback_result.strip():
                # 最后的回退：生成一个简单日志查询
                fallback_result = f"""
SELECT log_id, timestamp, source_ip, event_type, message, detected_by 
FROM logs 
WHERE {self._get_time_condition_sql(time_range)}
ORDER BY timestamp DESC 
LIMIT 100
                """.strip()
            
            return {
                "sql": fallback_result,
                "confidence": 0.5,  # 回退方法的置信度较低
                "original_query": query_text,
                "error": str(e)
            }
    
    def _preprocess_query(self, query_text: str) -> Tuple[str, Dict[str, Any]]:
        """
        预处理查询文本，提取时间范围、实体等信息
        
        Args:
            query_text: 原始查询文本
            
        Returns:
            处理后的查询文本和时间范围信息的元组
        """
        # 初始化时间范围字典
        time_range = {
            "type": "relative",  # 相对时间还是绝对时间范围
            "unit": "hour",      # 时间单位：hour, day, week, month
            "value": 24,         # 默认查询最近24小时
            "start": None,       # 绝对时间范围的开始时间
            "end": None          # 绝对时间范围的结束时间
        }
        
        # 处理时间表达式
        # 1. 相对时间表达式
        time_patterns = [
            # 最近X小时/天/周/月
            (r'(最近|过去)(\d+)(小时|天|周|月)', 
             lambda m: self._update_time_range(time_range, m.group(2), m.group(3))),
            # 今天/昨天/本周/上周/本月/上个月
            (r'(今天|昨天|本周|上周|本月|上个月)', 
             lambda m: self._handle_special_time_range(time_range, m.group(1))),
            # X小时/天前
            (r'(\d+)(小时|天)前', 
             lambda m: self._update_time_range(time_range, m.group(1), m.group(2))),
            # 从X到Y（绝对时间范围）
            (r'从(.+?)到(.+?)(的|之间)?', 
             lambda m: self._parse_absolute_time_range(time_range, m.group(1), m.group(2)))
        ]
        
        processed_text = query_text
        for pattern, handler in time_patterns:
            match = re.search(pattern, processed_text)
            if match:
                handler(match)  # 更新时间范围
                processed_text = re.sub(pattern, '在指定时间范围内', processed_text)
        
        # 处理特定实体和关键词
        # 1. 用户名
        user_patterns = [
            (r'用户[是为:：\s]*([^\s,，;；]+)', r'user="\1"'),  # 用户是xxx
            (r'([^\s,，;；]+)用户', r'user="\1"')                # xxx用户
        ]
        
        for pattern, replacement in user_patterns:
            if re.search(pattern, processed_text):
                processed_text = re.sub(pattern, replacement, processed_text)
        
        # 2. IP地址
        ip_patterns = [
            (r'IP[是为:：\s]*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', r'source_ip="\1"'), # IP是xxx
            (r'来自[IP是为:：\s]*(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', r'source_ip="\1"') # 来自IP xxx
        ]
        
        for pattern, replacement in ip_patterns:
            if re.search(pattern, processed_text):
                processed_text = re.sub(pattern, replacement, processed_text)
        
        # 3. 事件类型关键词
        event_mappings = {
            '登录': 'login',
            '登陆': 'login',
            '注销': 'logout',
            '访问': 'access',
            '读取': 'read',
            '写入': 'write',
            '删除': 'delete',
            '修改': 'modify',
            '上传': 'upload',
            '下载': 'download',
            '执行': 'execute'
        }
        
        for cn_event, en_event in event_mappings.items():
            if cn_event in processed_text:
                processed_text = processed_text.replace(cn_event, f'event_type="{en_event}"')
        
        # 4. 状态关键词
        status_patterns = [
            (r'(成功|失败|拒绝|通过|阻止)', 
             lambda m: self._map_status_keyword(processed_text, m.group(1)))
        ]
        
        for pattern, handler in status_patterns:
            match = re.search(pattern, processed_text)
            if match:
                processed_text = handler(match)
        
        # 5. 异常关键词
        if re.search(r'(异常|可疑|攻击|威胁|恶意|风险)', processed_text):
            processed_text += ' detected_by!=none'
        
        # 6. 处理特定查询类型
        if '登录失败' in query_text or '失败登录' in query_text:
            processed_text += ' event_type="login" AND message LIKE "%failure%"'
        
        # 7. 处理排序和限制
        if '前' in processed_text and re.search(r'前(\d+)', processed_text):
            limit = re.search(r'前(\d+)', processed_text).group(1)
            processed_text = re.sub(r'前(\d+)', f'LIMIT {limit}', processed_text)
        
        # 8. 默认添加排序（按时间倒序）
        if not 'ORDER BY' in processed_text.upper():
            processed_text += ' ORDER BY timestamp DESC'
        
        # 9. 默认限制返回结果数量
        if not 'LIMIT' in processed_text.upper():
            processed_text += ' LIMIT 100'
        
        return processed_text, time_range
    
    def _update_time_range(self, time_range: Dict[str, Any], value: str, unit: str) -> None:
        """更新时间范围信息"""
        time_range["type"] = "relative"
        time_range["value"] = int(value)
        
        unit_mapping = {
            "小时": "hour",
            "天": "day",
            "周": "week",
            "月": "month"
        }
        
        time_range["unit"] = unit_mapping.get(unit, "hour")
    
    def _handle_special_time_range(self, time_range: Dict[str, Any], time_expr: str) -> None:
        """处理特殊时间表达式如今天、昨天等"""
        time_range["type"] = "special"
        
        now = datetime.now()
        
        if time_expr == "今天":
            time_range["start"] = datetime(now.year, now.month, now.day, 0, 0, 0)
            time_range["end"] = datetime(now.year, now.month, now.day, 23, 59, 59)
        elif time_expr == "昨天":
            yesterday = now - timedelta(days=1)
            time_range["start"] = datetime(yesterday.year, yesterday.month, yesterday.day, 0, 0, 0)
            time_range["end"] = datetime(yesterday.year, yesterday.month, yesterday.day, 23, 59, 59)
        elif time_expr == "本周":
            start_of_week = now - timedelta(days=now.weekday())
            time_range["start"] = datetime(start_of_week.year, start_of_week.month, start_of_week.day, 0, 0, 0)
            time_range["end"] = now
        elif time_expr == "上周":
            start_of_last_week = now - timedelta(days=now.weekday() + 7)
            end_of_last_week = now - timedelta(days=now.weekday() + 1)
            time_range["start"] = datetime(start_of_last_week.year, start_of_last_week.month, start_of_last_week.day, 0, 0, 0)
            time_range["end"] = datetime(end_of_last_week.year, end_of_last_week.month, end_of_last_week.day, 23, 59, 59)
        elif time_expr == "本月":
            time_range["start"] = datetime(now.year, now.month, 1, 0, 0, 0)
            time_range["end"] = now
        elif time_expr == "上个月":
            if now.month == 1:
                last_month_year = now.year - 1
                last_month = 12
            else:
                last_month_year = now.year
                last_month = now.month - 1
                
            time_range["start"] = datetime(last_month_year, last_month, 1, 0, 0, 0)
            
            # 计算上个月最后一天
            if last_month == 12:
                next_month_year = last_month_year + 1
                next_month = 1
            else:
                next_month_year = last_month_year
                next_month = last_month + 1
                
            last_day_of_month = (datetime(next_month_year, next_month, 1) - timedelta(days=1)).day
            time_range["end"] = datetime(last_month_year, last_month, last_day_of_month, 23, 59, 59)
    
    def _parse_absolute_time_range(self, time_range: Dict[str, Any], start_str: str, end_str: str) -> None:
        """解析绝对时间范围"""
        time_range["type"] = "absolute"
        
        # 简单处理，实际中可能需要更复杂的日期解析
        try:
            # 尝试各种常见的日期格式
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d %H:%M",
                "%Y/%m/%d",
                "%Y年%m月%d日 %H:%M:%S",
                "%Y年%m月%d日 %H:%M",
                "%Y年%m月%d日",
                "%m月%d日 %H:%M"
            ]
            
            start_time = None
            end_time = None
            
            for fmt in formats:
                try:
                    start_time = datetime.strptime(start_str, fmt)
                    break
                except ValueError:
                    continue
            
            for fmt in formats:
                try:
                    end_time = datetime.strptime(end_str, fmt)
                    break
                except ValueError:
                    continue
            
            if start_time:
                time_range["start"] = start_time
            
            if end_time:
                time_range["end"] = end_time
            
        except Exception:
            # 如果解析失败，回退到默认时间范围
            time_range["type"] = "relative"
            time_range["unit"] = "hour"
            time_range["value"] = 24
    
    def _map_status_keyword(self, text: str, status_word: str) -> str:
        """映射状态关键词"""
        status_mapping = {
            "成功": "success",
            "失败": "failure",
            "拒绝": "denied",
            "通过": "success",
            "阻止": "blocked"
        }
        
        if status_word in status_mapping:
            return text.replace(status_word, f'message LIKE "%{status_mapping[status_word]}%"')
        
        return text
    
    def _build_prompt(self, query_text: str, time_range: Dict[str, Any]) -> str:
        """
        构建提示
        
        Args:
            query_text: 处理后的查询文本
            time_range: 时间范围信息
        """
        # 表结构字符串
        schema_str = ""
        for table_name, table_info in self.table_schema.items():
            schema_str += f"表名: {table_name}\n"
            schema_str += "列:\n"
            for column in table_info["columns"]:
                schema_str += f"- {column['name']} ({column['type']}): {column['description']}\n"
        
        # 构建时间条件
        time_condition = self._get_time_condition_sql(time_range)
        
        # 构建完整提示
        prompt = f"""
你是一个专门将自然语言转换为SQL查询的AI助手。根据以下数据库表结构、查询和时间范围，生成一个或多个(多个查询使用Union连接)有效的SQL查询语句。

数据库结构:
{schema_str}

用户查询: {query_text}

时间范围条件（如适用）: {time_condition}

请根据用户的需求，按照以下指南生成SQL查询:

1. 主要查询：针对用户的主要需求生成一个主SQL查询
2. 补充查询（可选）：如果单个查询无法满足用户的所有需求，请生成额外的SQL查询
3. 合并查询（可选）：如有必要，使用UNION或子查询合并多个查询结果

LIKE语法增强指南:
- 使用多个LIKE条件并用OR连接，以匹配不同格式的相同信息
- 对于用户名，考虑匹配: 'username=X', 'user=X', 'X_user', 'user:X', '用户=X', 'user_name=X', 'userid=X'等
- 对于IP地址，考虑匹配: 'IP=X', 'source=X', 'from X', 'src=X', 'source_ip=X', 'ip_addr=X'等
- 对于事件类型，考虑匹配: 'event=X', 'action=X', 'operation=X', 'type=X'等
- 对于登录失败，考虑匹配: 'login failed', 'auth failure', 'authentication failed', 'invalid credentials'等
- 考虑字段间的空格、标点符号、大小写变化、英语和中文、及不同日志格式的差异

只返回SQL查询语句，不需要任何其他解释。确保SQL语句对SQLite是有效的。
如果查询涉及日志数据，默认查询logs表；如果涉及异常检测结果，查询anomalies表；如果涉及规则，查询rules表。
可以使用JOIN语句关联多个表。确保生成的SQL语法正确且能正确执行。
"""
        
        return prompt
    
    def _get_time_condition_sql(self, time_range: Dict[str, Any]) -> str:
        """根据时间范围生成SQL条件"""
        if time_range["type"] == "relative":
            unit = time_range["unit"]
            value = time_range["value"]
            
            unit_mapping = {
                "hour": "hour",
                "day": "day",
                "week": "day",  # SQLite没有直接的周单位，转换为天
                "month": "month"
            }
            
            # 处理周的特殊情况
            if unit == "week":
                value = value * 7  # 转换为天
            
            return f"timestamp >= datetime('now', '-{value} {unit_mapping[unit]}')"
        
        elif time_range["type"] == "special" or time_range["type"] == "absolute":
            conditions = []
            
            if time_range["start"]:
                start_str = time_range["start"].strftime("%Y-%m-%d %H:%M:%S")
                conditions.append(f"timestamp >= '{start_str}'")
            
            if time_range["end"]:
                end_str = time_range["end"].strftime("%Y-%m-%d %H:%M:%S")
                conditions.append(f"timestamp <= '{end_str}'")
            
            if conditions:
                return " AND ".join(conditions)
        
        # 默认返回最近24小时
        return "timestamp >= datetime('now', '-24 hour')"
    
    def _call_openrouter_api(self, prompt: str) -> str:
        """调用OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "你是一个专门将自然语言转换为SQL查询的AI助手。仔细理解用户需求，生成适当的SQL查询语句。如果需要多个查询，优先使用UNION合并它们，确保合并查询的列名和类型保持一致。只输出SQL查询语句，不要加任何解释。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_price": {
                "prompt": 0,
                "completion": 0
            }
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
        
        # 如果没有代码块，匹配单个完整的SQL查询（可能包含UNION或多行）
        # 优先匹配带有UNION的复杂查询
        union_sql_match = re.search(r'(SELECT[\s\S]*UNION[\s\S]*?;)', text, re.IGNORECASE)
        if union_sql_match:
            return union_sql_match.group(1).strip()
        
        # 匹配多个SQL语句（以分号结尾）- 作为后备方案
        sql_matches = re.findall(r'(SELECT[\s\S]*?;)', text)
        if sql_matches and len(sql_matches) > 1:
            # 如果有多个SQL语句，尝试将它们通过UNION合并
            # 首先检查是否每个查询都有相同的列数（简单检查）
            return "\n\nUNION\n\n".join(item.strip() for item in sql_matches)
        elif sql_matches and len(sql_matches) == 1:
            return sql_matches[0].strip()
        
        # 匹配单个没有分号的SQL语句（兼容旧格式）
        sql_match = re.search(r'(SELECT[\s\S]*?)(?:\n|$)', text)
        if sql_match:
            return sql_match.group(1).strip()
        
        # 如果上述都没匹配到，返回整个文本
        return text.strip()
    
    def _enhance_entity_matching(self, sql: str, processed_query: str) -> str:
        """
        增强SQL中的实体匹配，支持多种匹配模式
        
        Args:
            sql: 原始SQL查询或带有UNION的复合查询
            processed_query: 处理后的查询文本
            
        Returns:
            增强后的SQL查询
        """
        # 检查是否有UNION连接的多个查询
        if re.search(r'\bUNION\b', sql, re.IGNORECASE):
            # 使用正则表达式拆分UNION查询，同时保留UNION关键字
            parts = re.split(r'\b(UNION(?:\s+ALL)?)\b', sql, flags=re.IGNORECASE)
            
            # 分离SQL部分和UNION关键字
            sql_parts = []
            union_parts = []
            
            for i, part in enumerate(parts):
                if i % 2 == 0:  # 偶数索引是SQL语句部分
                    sql_parts.append(part)
                else:  # 奇数索引是UNION关键字
                    union_parts.append(part)
            
            # 对每个SQL部分单独进行增强
            enhanced_parts = []
            for part in sql_parts:
                if part.strip():  # 确保不是空字符串
                    enhanced_part = self._enhance_single_query(part.strip(), processed_query)
                    enhanced_parts.append(enhanced_part)
            
            # 重新组合增强后的查询和UNION关键字
            result = ""
            for i in range(len(enhanced_parts)):
                result += enhanced_parts[i]
                if i < len(union_parts):
                    result += "\n" + union_parts[i] + "\n"
            
            return result
        
        # 检查是否有分号分隔的多个查询（旧的处理方式，作为后备）
        elif ";" in sql and "\n" in sql:
            sql_queries = sql.split(";")
            # 移除空白查询
            sql_queries = [q.strip() for q in sql_queries if q.strip()]
            # 对每个查询单独进行增强
            enhanced_queries = []
            for query in sql_queries:
                enhanced_query = self._enhance_single_query(query, processed_query)
                if enhanced_query:
                    enhanced_queries.append(enhanced_query)
            
            # 尝试用UNION合并查询
            if len(enhanced_queries) > 1:
                return "\n\nUNION\n\n".join(enhanced_queries)
            elif enhanced_queries:
                return enhanced_queries[0]
            else:
                return ""
        else:
            # 单个查询的情况
            return self._enhance_single_query(sql, processed_query)
    
    def _enhance_single_query(self, sql: str, processed_query: str) -> str:
        """
        增强单个SQL查询
        
        Args:
            sql: 原始SQL查询
            processed_query: 处理后的查询文本
            
        Returns:
            增强后的SQL查询
        """
        if not sql.strip():
            return sql
            
        # 提取查询中的关键实体和条件
        # 1. 用户名匹配
        users = []
        user_match = re.search(r'user\s*=\s*[\'"]([^\'"]+)[\'"]', sql, re.IGNORECASE)
        if user_match:
            users.append(user_match.group(1))
        
        # 检查查询文本中的常见用户名
        common_users = ["admin", "root", "administrator", "system", "guest", "user"]
        for user in common_users:
            if user in processed_query.lower() and user not in users:
                users.append(user)
        
        # 2. IP地址匹配
        ips = []
        ip_matches = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', processed_query)
        if ip_matches:
            ips.extend(ip_matches)
        
        ip_match = re.search(r'source_ip\s*=\s*[\'"]([^\'"]+)[\'"]', sql, re.IGNORECASE)
        if ip_match and ip_match.group(1) not in ips:
            ips.append(ip_match.group(1))
        
        # 3. 事件类型匹配
        events = []
        event_match = re.search(r'event_type\s*=\s*[\'"]([^\'"]+)[\'"]', sql, re.IGNORECASE)
        if event_match:
            events.append(event_match.group(1))
        
        # 检查常见事件类型
        common_events = {
            "login": ["登录", "登陆", "signin", "sign-in", "log in"],
            "logout": ["登出", "注销", "signout", "sign-out", "log out"],
            "access": ["访问", "access", "visit"],
            "file_access": ["文件访问", "file access", "读取文件", "写入文件"],
            "failed_login": ["登录失败", "login failed", "认证失败"],
            "authentication": ["认证", "鉴权", "auth"]
        }
        
        for event, keywords in common_events.items():
            if any(keyword in processed_query.lower() for keyword in keywords) and event not in events:
                events.append(event)
        
        # 4. 检测状态匹配
        statuses = []
        status_patterns = [
            (r'failed', ["失败", "错误", "拒绝", "无效"]),
            (r'success', ["成功", "通过", "允许"]),
            (r'denied', ["拒绝", "禁止", "阻止"]),
            (r'error', ["错误", "异常", "失败"])
        ]
        
        for status, keywords in status_patterns:
            if any(keyword in processed_query.lower() for keyword in keywords):
                statuses.append(status)
        
        # 5. 应用增强匹配
        enhanced_sql = sql
        
        # 处理用户名
        for user in users:
            user_matches = self._generate_multiple_matches("user", user)
            simple_user_pattern = fr"user\s*=\s*['\"]?{user}['\"]?"
            if re.search(simple_user_pattern, enhanced_sql, re.IGNORECASE):
                user_condition = f"({' OR '.join(user_matches)})"
                enhanced_sql = re.sub(simple_user_pattern, user_condition, enhanced_sql, flags=re.IGNORECASE)
        
        # 处理IP地址
        for ip in ips:
            ip_matches = self._generate_multiple_matches("source_ip", ip)
            simple_ip_pattern = fr"source_ip\s*=\s*['\"]?{ip}['\"]?"
            if re.search(simple_ip_pattern, enhanced_sql, re.IGNORECASE):
                ip_condition = f"({' OR '.join(ip_matches)})"
                enhanced_sql = re.sub(simple_ip_pattern, ip_condition, enhanced_sql, flags=re.IGNORECASE)
        
        # 处理事件类型
        for event in events:
            event_matches = self._generate_multiple_matches("event_type", event)
            simple_event_pattern = fr"event_type\s*=\s*['\"]?{event}['\"]?"
            if re.search(simple_event_pattern, enhanced_sql, re.IGNORECASE):
                event_condition = f"({' OR '.join(event_matches)})"
                enhanced_sql = re.sub(simple_event_pattern, event_condition, enhanced_sql, flags=re.IGNORECASE)
        
        # 增强LIKE条件
        message_like_pattern = r"message\s+LIKE\s+'%([^%]+)%'"
        message_like_matches = re.findall(message_like_pattern, enhanced_sql, re.IGNORECASE)
        
        # 特殊处理登录失败情况 - 组合事件类型和状态
        if ("login" in events or "login" in enhanced_sql.lower()) and \
           (any(s in statuses for s in ["failed", "failure"]) or "fail" in enhanced_sql.lower()):
            if not re.search(r"login\s+failed", enhanced_sql.lower()):
                # 添加登录失败相关条件
                login_fail_conditions = [
                    "message LIKE '%login failed%'",
                    "message LIKE '%failed login%'",
                    "message LIKE '%login failure%'",
                    "message LIKE '%authentication failed%'",
                    "message LIKE '%auth failure%'",
                    "message LIKE '%登录失败%'",
                    "message LIKE '%认证失败%'"
                ]
                
                # 找到WHERE子句的位置，添加额外条件
                where_match = re.search(r'\bWHERE\b', enhanced_sql, re.IGNORECASE)
                if where_match:
                    # 添加到WHERE后的条件中
                    position = where_match.end()
                    condition_str = f" ({' OR '.join(login_fail_conditions)}) AND"
                    enhanced_sql = enhanced_sql[:position] + condition_str + enhanced_sql[position:]
        
        # 处理一般LIKE条件
        if message_like_matches:
            for term in message_like_matches:
                enhanced_like_conditions = []
                # 添加原始条件
                enhanced_like_conditions.append(f"message LIKE '%{term}%'")
                
                # 根据不同的术语添加变体
                term_lower = term.lower()
                
                # 用户名相关变体
                if term_lower in ["admin", "administrator", "root", "system", "guest"]:
                    user_matches = self._generate_multiple_matches("user", term)
                    enhanced_like_conditions.extend(user_matches)
                
                # 登录状态相关变体
                elif term_lower in ["failed", "failure", "unsuccessful", "error"]:
                    failure_patterns = [
                        f"login {term_lower}", f"{term_lower} login", 
                        f"authentication {term_lower}", f"{term_lower} authentication",
                        f"access {term_lower}", f"{term_lower} access"
                    ]
                    for pattern in failure_patterns:
                        enhanced_like_conditions.append(f"message LIKE '%{pattern}%'")
                
                # 通用变体（处理空格、分隔符等）
                enhanced_like_conditions.append(f"message LIKE '%{term.replace(' ', '')}%'")  # 移除空格
                enhanced_like_conditions.append(f"message LIKE '%{term.replace(' ', '_')}%'")  # 将空格替换为下划线
                enhanced_like_conditions.append(f"message LIKE '%{term.replace('=', ':')}%'")  # 将等号替换为冒号
                enhanced_like_conditions.append(f"LOWER(message) LIKE '%{term_lower}%'")  # 不区分大小写
                
                if len(enhanced_like_conditions) > 1:
                    # 替换原始条件
                    original_condition = f"message LIKE '%{term}%'"
                    new_condition = f"({' OR '.join(enhanced_like_conditions)})"
                    enhanced_sql = enhanced_sql.replace(original_condition, new_condition)
        
        return enhanced_sql
    
    def _generate_multiple_matches(self, field: str, value: str) -> List[str]:
        """
        为字段生成多种匹配条件
        
        Args:
            field: 字段名称
            value: 字段值
            
        Returns:
            包含多种匹配条件的列表
        """
        matches = []
        
        # 精确匹配
        matches.append(f"{field} = '{value}'")
        matches.append(f"{field} LIKE '{value}'")
        
        # 通用模式匹配
        matches.append(f"message LIKE '%{value}%'")
        
        # 如果是用户字段
        if field == "user":
            # 常见用户名模式
            user_patterns = [
                f"user={value}", f"username={value}", f"{value}_user", 
                f"user:{value}", f"用户={value}", f"user_name={value}", 
                f"userid={value}", f"user_id={value}", f"usr={value}",
                f"login={value}", f"account={value}", f"id={value}",
                f"auth={value}", f"authentication={value}", f"user '{value}'",
                f"user \"{value}\"", f"user\\t{value}", f"user\\n{value}"
            ]
            
            for pattern in user_patterns:
                matches.append(f"message LIKE '%{pattern}%'")
            
            # 不区分大小写的变体
            matches.append(f"LOWER(message) LIKE '%user={value.lower()}%'")
            matches.append(f"LOWER(message) LIKE '%username={value.lower()}%'")
            
            # 处理包含空格和标点符号的变体
            matches.append(f"message LIKE '%user = {value}%'")
            matches.append(f"message LIKE '%user= {value}%'")
            matches.append(f"message LIKE '%user ={value}%'")
            matches.append(f"message LIKE '%user: {value}%'")
        
        # 如果是IP字段
        elif field == "source_ip":
            # 常见IP模式
            ip_patterns = [
                f"IP={value}", f"source={value}", f"from {value}", 
                f"src={value}", f"source_ip={value}", f"ip_addr={value}",
                f"addr={value}", f"address={value}", f"来源={value}",
                f"源IP={value}", f"destination={value}", f"dst={value}",
                f"ip:{value}", f"host={value}", f"client={value}",
                f"source_address={value}", f"client_ip={value}"
            ]
            
            for pattern in ip_patterns:
                matches.append(f"message LIKE '%{pattern}%'")
            
            # 处理包含空格和标点符号的变体
            matches.append(f"message LIKE '%IP : {value}%'")
            matches.append(f"message LIKE '%IP: {value}%'")
            matches.append(f"message LIKE '%IP = {value}%'")
            matches.append(f"message LIKE '%IP= {value}%'")
            matches.append(f"message LIKE '%IP ={value}%'")
        
        # 如果是事件类型字段
        elif field == "event_type":
            # 常见事件类型模式
            event_patterns = [
                f"event={value}", f"action={value}", f"operation={value}", 
                f"type={value}", f"event_type={value}", f"activity={value}",
                f"事件={value}", f"操作={value}", f"行为={value}",
                f"{value} event", f"{value} operation", f"event:{value}",
                f"action:{value}", f"type:{value}"
            ]
            
            for pattern in event_patterns:
                matches.append(f"message LIKE '%{pattern}%'")
            
            # 特殊处理登录事件
            if value.lower() == "login":
                login_patterns = [
                    "login", "logged in", "sign in", "signin", "log in", 
                    "logging in", "登录", "登陆", "认证", "鉴权"
                ]
                
                for pattern in login_patterns:
                    matches.append(f"message LIKE '%{pattern}%'")
            
            # 特殊处理失败状态
            elif value.lower() == "failure" or value.lower() == "failed":
                failure_patterns = [
                    "failed", "failure", "unsuccessful", "denied", "rejected",
                    "invalid", "wrong", "incorrect", "error", "失败", "拒绝", 
                    "错误", "无效"
                ]
                
                for pattern in failure_patterns:
                    matches.append(f"message LIKE '%{pattern}%'")
        
        return matches
    
    def _pattern_matching_fallback(self, query_text: str, time_range: Dict[str, Any]) -> str:
        """
        模式匹配回退方法，当API调用失败时使用
        
        Args:
            query_text: 处理后的查询文本
            time_range: 时间范围信息
        
        Returns:
            生成的SQL查询或多个查询合并后的结果
        """
        # 生成时间条件
        time_condition = self._get_time_condition_sql(time_range)
        
        # 提取关键信息
        # 用户名模式
        user_match = re.search(r'user="([^"]+)"', query_text)
        user_condition = ""
        if user_match:
            user = user_match.group(1)
            # 使用多种匹配模式
            user_matches = self._generate_multiple_matches("user", user)
            user_condition = f"AND ({' OR '.join(user_matches)}) "
        
        # IP地址模式
        ip_match = re.search(r'source_ip="([^"]+)"', query_text)
        ip_condition = ""
        if ip_match:
            ip = ip_match.group(1)
            # 使用多种匹配模式
            ip_matches = self._generate_multiple_matches("source_ip", ip)
            ip_condition = f"AND ({' OR '.join(ip_matches)}) "
        
        # 处理可能存在的简单用户名模式 (如 "admin登录失败")
        if not user_match and "admin" in query_text:
            user_matches = self._generate_multiple_matches("user", "admin")
            user_condition = f"AND ({' OR '.join(user_matches)}) "
        
        # 事件类型模式
        event_match = re.search(r'event_type="([^"]+)"', query_text)
        event_condition = ""
        if event_match:
            event = event_match.group(1)
            event_condition = f"AND event_type = '{event}' "
        
        # 状态条件
        status_match = re.search(r'message LIKE "%([^%]+)%"', query_text)
        status_condition = ""
        if status_match:
            status = status_match.group(1)
            status_condition = f"AND message LIKE '%{status}%' "
        
        # 创建查询结果列表和它们的相关性分数
        query_relevance = []
        
        # 检测登录失败相关查询
        if ('登录失败' in query_text or '失败登录' in query_text or 
            (event_condition.find('login') > -1 and status_condition.find('failure') > -1)):
            login_failures_query = f"""
SELECT 
    COUNT(*) as count, 
    user as entity_name, 
    source_ip as source, 
    'login_failure' as event_category,
    'count' as data_type,
    timestamp as event_time
FROM logs 
WHERE {time_condition} 
{event_condition}
{status_condition}
{user_condition}
{ip_condition}
AND (message LIKE '%fail%' OR message LIKE '%失败%' OR message LIKE '%拒绝%' OR message LIKE '%deny%' OR message LIKE '%invalid%')
GROUP BY user, source_ip 
ORDER BY count DESC
            """.strip()
            relevance = 8 if ('登录失败' in query_text or '失败登录' in query_text) else 5
            query_relevance.append((login_failures_query, relevance))
        
        # 检测异常查询
        if '异常' in query_text or '可疑' in query_text or '攻击' in query_text or 'detected_by!=none' in query_text:
            anomalies_query = f"""
SELECT 
    a.anomaly_id as id, 
    l.source_ip as source, 
    a.anomaly_type as event_category,
    'anomaly' as data_type,
    l.timestamp as event_time,
    l.message as description
FROM anomalies a
JOIN logs l ON a.log_id = l.log_id 
WHERE {time_condition} 
{user_condition}
{ip_condition}
{event_condition}
ORDER BY l.timestamp DESC
            """.strip()
            relevance = 7 if any(term in query_text for term in ['异常', '可疑', '攻击']) else 4
            query_relevance.append((anomalies_query, relevance))
        
        # 检测规则查询
        if '规则' in query_text:
            rules_query = f"""
SELECT 
    rule_id as id, 
    name as entity_name, 
    '' as source,
    'rule' as data_type,
    action as event_category,
    created_at as event_time
FROM rules 
ORDER BY created_at DESC
            """.strip()
            relevance = 6 if '规则' in query_text else 3
            query_relevance.append((rules_query, relevance))
        
        # 通用日志查询（总是添加，作为默认选项）
        general_logs_query = f"""
SELECT 
    log_id as id, 
    source_ip as source, 
    event_type as event_category,
    'log' as data_type,
    timestamp as event_time,
    message as description
FROM logs 
WHERE {time_condition} 
{user_condition}
{ip_condition}
{event_condition}
{status_condition}
ORDER BY timestamp DESC 
LIMIT 100
            """.strip()
        
        # 如果查询是关于日志的一般查询，提高相关性
        relevance = 2  # 默认相关性较低
        if any(term in query_text for term in ['日志', '记录', 'log', 'record', '列出']):
            relevance = 5
        
        query_relevance.append((general_logs_query, relevance))
        
        # 按相关性排序查询
        query_relevance.sort(key=lambda x: x[1], reverse=True)
        
        # 提取排序后的查询
        queries = [q[0] for q in query_relevance]
        
        # 如果只有一个查询，直接返回
        if len(queries) == 1:
            return queries[0]
        
        # 检查是否有特定关键词要求合并结果
        merge_keywords = ['合并', '综合', '全部', '所有', 'all', 'combine', 'merge', 'union']
        should_merge = any(keyword in query_text.lower() for keyword in merge_keywords)
        
        if should_merge:
            # 生成合并查询
            return f"""
/* 第一个查询（主要相关） */
{queries[0]}

UNION

/* 第二个查询（次要相关） */
{queries[1] if len(queries) > 1 else ''}
            """.strip()
        else:
            # 如果没有合并关键词，返回最相关的查询
            return queries[0]


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
