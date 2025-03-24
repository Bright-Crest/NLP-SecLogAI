#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志数据集
用于加载和处理安全日志数据，转换为模型输入
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LogDataset(Dataset):
    """安全日志数据集，用于BERT模型训练"""
    
    def __init__(self, data, tokenizer_name='bert-base-uncased', max_length=128):
        """
        初始化数据集
        
        Args:
            data: 包含日志文本和标签的字典列表，格式为[{'text': '...', 'label': 0}, ...]
            tokenizer_name: 分词器名称
            max_length: 序列最大长度
        """
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据集中的一项"""
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # 使用分词器处理文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除第一个维度，因为tokenizer返回batch
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 添加标签
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        
        return encoding


class AnomalyDetectionDataset(Dataset):
    """异常检测数据集，用于无监督学习"""
    
    def __init__(self, data, tokenizer_name='bert-base-uncased', max_length=128):
        """
        初始化数据集
        
        Args:
            data: 包含日志文本的字典列表，格式为[{'text': '...', 'is_anomaly': True}, ...]
            tokenizer_name: 分词器名称
            max_length: 序列最大长度
        """
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取数据集中的一项"""
        item = self.data[idx]
        text = item['text']
        is_anomaly = item.get('is_anomaly', False)
        
        # 使用分词器处理文本
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 移除第一个维度，因为tokenizer返回batch
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 添加异常标记
        encoding['is_anomaly'] = torch.tensor(1 if is_anomaly else 0, dtype=torch.float)
        
        return encoding 