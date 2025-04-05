import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.models.log_tokenizer import LogTokenizer


class LogWindow:
    """
    日志窗口处理类，用于构建日志行为序列
    支持固定窗口和滑动窗口两种模式
    """
    
    def __init__(self, tokenizer_name='prajjwal1/bert-mini', max_length=128, window_size=10):
        """
        初始化日志窗口处理器
        
        参数:
            tokenizer_name: 使用的tokenizer名称
            max_length: 每个窗口的最大token长度
            window_size: 窗口大小(每个窗口包含的日志行数)
        """
        self.log_tokenizer = LogTokenizer(tokenizer_name)
        self.window_size = window_size
        self.max_length = max_length
        self.sep_token = self.log_tokenizer.tokenizer.sep_token
    
    def create_fixed_windows(self, log_list):
        """
        将日志列表按固定窗口大小分组
        
        参数:
            log_list: 日志文本列表
            
        返回:
            window_tokens: 窗口token列表
            remaining_logs: 剩余未处理日志(不足一个窗口)
        """
        window_tokens = []
        
        # 按固定窗口大小分组
        for i in range(0, len(log_list), self.window_size):
            window_logs = log_list[i:i + self.window_size]
            
            # 若最后一组不足window_size且不需处理，则返回剩余日志
            if len(window_logs) < self.window_size and i + self.window_size < len(log_list):
                return window_tokens, log_list[i:]
            
            # 将窗口内日志用[SEP]连接
            window_text = f" {self.sep_token} ".join(window_logs)
            tokens = self.log_tokenizer.tokenize(window_text)
            window_tokens.append(tokens)
        
        return window_tokens, []  # 返回所有窗口的token和空的剩余日志
    
    def create_sliding_windows(self, log_list, stride=1):
        """
        将日志列表按滑动窗口处理
        
        参数:
            log_list: 日志文本列表
            stride: 滑动步长
            
        返回:
            window_tokens: 窗口token列表
        """
        window_tokens = []
        
        # 确保日志数量足够处理
        if len(log_list) < self.window_size:
            return window_tokens
        
        # 按滑动窗口处理
        for i in range(0, len(log_list) - self.window_size + 1, stride):
            window_logs = log_list[i:i + self.window_size]
            window_text = f" {self.sep_token} ".join(window_logs)
            tokens = self.log_tokenizer.tokenize(window_text)
            window_tokens.append(tokens)
        
        return window_tokens
    
    def batch_windows(self, window_tokens):
        """
        将窗口token列表转换为批量tensor
        
        参数:
            window_tokens: 窗口token列表
            
        返回:
            batch: 包含input_ids和attention_mask的字典
        """
        input_ids = []
        attention_masks = []
        
        for tokens in window_tokens:
            input_ids.append(tokens['input_ids'])
            attention_masks.append(tokens['attention_mask'])
        
        # 堆叠tensor
        if input_ids:
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_masks
            }
        else:
            return None 