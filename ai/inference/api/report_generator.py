#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
安全报告生成器
生成基于日志分析的安全事件报告，包括异常检测和修复建议
"""

import os
import re
import json
import datetime
from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
import openai

# 加载OpenAI API密钥
openai.api_key = os.environ.get("OPENAI_API_KEY", "")


class ReportGenerator:
    """安全报告生成器"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", max_tokens: int = 1000):
        """
        初始化报告生成器
        
        Args:
            model_name: OpenAI模型名称
            max_tokens: 最大生成令牌数
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def generate(self, logs: List[Dict[str, Any]], time_range: str = "24h") -> str:
        """
        生成安全报告
        
        Args:
            logs: 日志列表
            time_range: 时间范围（例如 '24h', '7d'）
            
        Returns:
            Markdown格式的安全报告
        """
        # 解析时间范围
        hours = self._parse_time_range(time_range)
        time_description = self._format_time_description(hours)
        
        # 提取关键统计信息
        stats = self._extract_statistics(logs)
        
        # 如果有OpenAI API密钥，使用GPT生成报告
        if openai.api_key:
            return self._generate_with_gpt(logs, stats, time_description)
        else:
            # 否则使用模板生成简单报告
            return self._generate_template_report(logs, stats, time_description)
    
    def _parse_time_range(self, time_range: str) -> int:
        """解析时间范围为小时数"""
        if time_range.endswith('h'):
            return int(time_range[:-1])
        elif time_range.endswith('d'):
            return int(time_range[:-1]) * 24
        elif time_range.endswith('w'):
            return int(time_range[:-1]) * 24 * 7
        else:
            return 24  # 默认24小时
    
    def _format_time_description(self, hours: int) -> str:
        """将小时数格式化为可读时间描述"""
        if hours == 24:
            return "过去24小时"
        elif hours < 24:
            return f"过去{hours}小时"
        elif hours % 24 == 0:
            days = hours // 24
            return f"过去{days}天"
        else:
            return f"过去{hours}小时"
    
    def _extract_statistics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从日志中提取统计信息"""
        stats = {
            "total_logs": len(logs),
            "anomaly_count": 0,
            "top_event_types": {},
            "top_sources": {},
            "top_users": {},
            "top_ips": {},
            "hourly_distribution": {}
        }
        
        # 创建计数器
        event_counter = Counter()
        source_counter = Counter()
        user_counter = Counter()
        ip_counter = Counter()
        hour_counter = Counter()
        
        # 收集异常
        anomalies = []
        
        # 处理每个日志
        for log in logs:
            # 计数异常
            if log.get("is_anomaly", False):
                stats["anomaly_count"] += 1
                anomalies.append(log)
            
            # 事件类型
            event_type = log.get("event_type", "unknown")
            event_counter[event_type] += 1
            
            # 来源
            source = log.get("source", "unknown")
            source_counter[source] += 1
            
            # 用户
            user = log.get("user")
            if user:
                user_counter[user] += 1
            
            # IP
            src_ip = log.get("src_ip")
            if src_ip:
                ip_counter[src_ip] += 1
            
            # 小时分布
            try:
                if isinstance(log.get("timestamp"), str):
                    # 从字符串解析
                    timestamp = datetime.datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
                    hour = timestamp.hour
                else:
                    # 假设是数字时间戳
                    timestamp = datetime.datetime.fromtimestamp(log["timestamp"])
                    hour = timestamp.hour
                
                hour_counter[hour] += 1
            except (KeyError, ValueError, TypeError):
                pass
        
        # 获取前5个最常见的事件类型、来源、用户和IP
        stats["top_event_types"] = dict(event_counter.most_common(5))
        stats["top_sources"] = dict(source_counter.most_common(5))
        stats["top_users"] = dict(user_counter.most_common(5))
        stats["top_ips"] = dict(ip_counter.most_common(5))
        
        # 按小时排序
        stats["hourly_distribution"] = {str(h): hour_counter[h] for h in sorted(hour_counter.keys())}
        
        # 添加异常日志
        stats["anomalies"] = anomalies[:10]  # 最多10条
        
        return stats
    
    def _generate_with_gpt(self, logs: List[Dict[str, Any]], stats: Dict[str, Any], time_description: str) -> str:
        """使用GPT生成报告"""
        # 准备提示
        prompt = f"""
作为安全分析师，请根据以下安全日志统计信息生成一份安全报告。报告应包括摘要、主要发现、异常检测结果和安全建议。

时间范围: {time_description}

日志统计信息:
- 总日志数: {stats['total_logs']}
- 异常日志数: {stats['anomaly_count']}
- 主要事件类型: {stats['top_event_types']}
- 主要来源: {stats['top_sources']}
- 活跃用户: {stats['top_users']}
- 主要IP: {stats['top_ips']}
- 每小时分布: {stats['hourly_distribution']}

异常日志样本:
"""
        
        # 添加最多5条异常日志
        for i, anomaly in enumerate(stats.get('anomalies', [])[:5]):
            prompt += f"\n{i+1}. {anomaly.get('raw_text', 'unknown')}"
        
        prompt += """

请生成一份包含以下部分的安全报告（使用Markdown格式）:
1. 安全事件摘要
2. 关键发现（突出异常与模式）
3. 主要安全威胁
4. 建议的修复措施
5. 后续步骤
"""
        
        try:
            # 调用API
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一名网络安全专家，负责分析安全日志并生成简洁但信息丰富的安全报告。使用清晰的Markdown格式。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=0.7
            )
            
            # 返回生成的报告
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            # 如果API调用失败，返回模板报告
            print(f"GPT报告生成失败，使用模板替代: {str(e)}")
            return self._generate_template_report(logs, stats, time_description)
    
    def _generate_template_report(self, logs: List[Dict[str, Any]], stats: Dict[str, Any], time_description: str) -> str:
        """使用模板生成报告"""
        # 基本报告模板
        report = f"""# 安全日志分析报告 - {time_description}

## 1. 安全事件摘要

- **总日志数**: {stats['total_logs']}
- **异常日志数**: {stats['anomaly_count']}
- **异常比例**: {(stats['anomaly_count'] / max(stats['total_logs'], 1)) * 100:.2f}%

## 2. 关键发现

### 主要事件类型
"""
        
        # 添加事件类型统计
        for event_type, count in stats['top_event_types'].items():
            report += f"- **{event_type}**: {count}条记录\n"
        
        report += "\n### 活跃来源\n"
        
        # 添加来源统计
        for source, count in stats['top_sources'].items():
            report += f"- **{source}**: {count}条记录\n"
        
        report += "\n### 活跃用户\n"
        
        # 添加用户统计
        for user, count in stats['top_users'].items():
            report += f"- **{user}**: {count}条记录\n"
        
        # 添加异常日志
        report += "\n## 3. 异常日志摘要\n\n"
        
        if stats.get('anomalies'):
            for i, anomaly in enumerate(stats['anomalies']):
                report += f"**{i+1}.** {anomaly.get('raw_text', 'unknown')}\n\n"
        else:
            report += "本报告期内未检测到异常。\n"
        
        # 添加安全建议
        report += """
## 4. 安全建议

1. **更新密码策略**: 确保所有用户使用强密码，并定期更换
2. **启用多因素认证**: 对关键账户启用MFA
3. **审查访问权限**: 检查用户权限，确保遵循最小权限原则
4. **监控异常活动**: 设置持续监控，及时发现可疑活动
5. **更新防火墙规则**: 限制对敏感系统的网络访问

## 5. 后续步骤

- 对异常行为进行深入调查
- 对发现的漏洞进行修复
- 加强安全意识培训
- 优化安全监控系统

*本报告由NLP-SecLogAI自动生成，建议安全团队进行进一步验证*
"""
        
        return report 