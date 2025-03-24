#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志预处理工具
包含日志清洗、解析和转换为训练数据的函数
"""

import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_log_text(text):
    """
    清洗原始日志文本
    
    Args:
        text: 原始日志文本
        
    Returns:
        清洗后的文本
    """
    # 移除特殊字符
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', text)
    
    # 标准化空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 移除可能的PII信息
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)  # IP地址
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)  # 邮箱
    
    return text


def parse_log_line(line, log_type):
    """
    解析单行日志
    
    Args:
        line: 单行日志文本
        log_type: 日志类型（如'windows'，'syslog'等）
        
    Returns:
        包含解析字段的字典
    """
    parsed = {'raw_text': line}
    
    if log_type == 'windows':
        # Windows事件日志解析
        if 'logon' in line.lower():
            parsed['event_type'] = 'logon'
            
            # 提取用户名
            user_match = re.search(r'user[:\s]+([^\s:]+)', line, re.IGNORECASE)
            if user_match:
                parsed['user'] = user_match.group(1)
            
            # 提取事件ID
            event_id_match = re.search(r'event id[:\s]+(\d+)', line, re.IGNORECASE)
            if event_id_match:
                parsed['event_id'] = event_id_match.group(1)
            
            # 检测失败/成功
            if 'fail' in line.lower() or 'error' in line.lower():
                parsed['status'] = 'failure'
            elif 'success' in line.lower():
                parsed['status'] = 'success'
    
    elif log_type == 'firewall':
        # 防火墙日志解析
        if 'blocked' in line.lower():
            parsed['event_type'] = 'connection_blocked'
            
            # 提取源IP
            src_ip_match = re.search(r'from\s+(\d+\.\d+\.\d+\.\d+)', line)
            if src_ip_match:
                parsed['src_ip'] = src_ip_match.group(1)
            
            # 提取目标IP
            dst_ip_match = re.search(r'to\s+(\d+\.\d+\.\d+\.\d+)', line)
            if dst_ip_match:
                parsed['dst_ip'] = dst_ip_match.group(1)
            
            # 提取端口
            port_match = re.search(r'port\s+(\d+)', line)
            if port_match:
                parsed['port'] = port_match.group(1)
    
    return parsed


def extract_features(parsed_logs):
    """
    从解析后的日志中提取特征
    
    Args:
        parsed_logs: 解析后的日志列表
        
    Returns:
        特征DataFrame
    """
    features = []
    
    for log in parsed_logs:
        feature = {}
        
        # 基本字段
        feature['text'] = log['raw_text']
        
        # 事件类型作为标签
        event_type_map = {
            'logon': 0,
            'logoff': 1,
            'connection_blocked': 2,
            'file_access': 3,
            'system_error': 4
        }
        feature['label'] = event_type_map.get(log.get('event_type'), 4)  # 默认为系统错误
        
        # 添加是否为异常的标记
        if log.get('status') == 'failure' or log.get('event_type') == 'connection_blocked':
            feature['is_anomaly'] = True
        else:
            feature['is_anomaly'] = False
        
        features.append(feature)
    
    return pd.DataFrame(features)


def preprocess_logs(log_path, val_split=0.2):
    """
    预处理日志文件并返回训练集和验证集
    
    Args:
        log_path: 日志文件路径（CSV或TXT）
        val_split: 验证集比例
        
    Returns:
        (训练集，验证集) 元组
    """
    # 读取日志文件
    if log_path.endswith('.csv'):
        # CSV格式，假设包含'text'和'label'列
        df = pd.read_csv(log_path)
        
    elif log_path.endswith('.txt'):
        # 纯文本，每行一条日志
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 解析日志
        parsed_logs = []
        for line in lines:
            line = line.strip()
            if line:
                # 根据日志内容猜测日志类型
                if 'windows' in line.lower() or 'event id' in line.lower():
                    log_type = 'windows'
                elif 'firewall' in line.lower() or 'blocked' in line.lower():
                    log_type = 'firewall'
                else:
                    log_type = 'generic'
                
                # 清洗并解析
                clean_line = clean_log_text(line)
                parsed = parse_log_line(clean_line, log_type)
                parsed_logs.append(parsed)
        
        # 提取特征
        df = extract_features(parsed_logs)
    
    else:
        # JSON格式，包含详细日志字段
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # 将JSON转换为DataFrame
        df = pd.json_normalize(logs)
    
    # 确保有必要的列
    required_cols = ['text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"日志数据缺少必要的列: {col}")
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['label'], random_state=42)
    
    # 转换为模型所需的列表格式
    train_data = train_df.to_dict('records')
    val_data = val_df.to_dict('records')
    
    return train_data, val_data


def balance_dataset(data, max_ratio=3.0):
    """
    平衡数据集，防止类别不平衡
    
    Args:
        data: 数据列表
        max_ratio: 允许的最大类别比率
        
    Returns:
        平衡后的数据集
    """
    # 计算每个类别的数量
    label_counts = {}
    for item in data:
        label = item['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 找出最小的类别
    min_count = min(label_counts.values())
    max_allowed = int(min_count * max_ratio)
    
    # 平衡数据集
    balanced_data = []
    label_current_counts = {label: 0 for label in label_counts.keys()}
    
    for item in data:
        label = item['label']
        if label_current_counts[label] < max_allowed:
            balanced_data.append(item)
            label_current_counts[label] += 1
    
    return balanced_data 