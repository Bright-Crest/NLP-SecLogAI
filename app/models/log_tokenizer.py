import re
from transformers import BertTokenizer

class LogTokenizer:
    def __init__(self, tokenizer_name='bert-base-uncased'):
        # 初始化BERT分词器
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        
    def preprocess_log(self, log_text):
        """预处理日志文本，移除时间戳、特殊字符等"""
        # 移除时间戳 (例如: "2023-10-10 08:02:30.123")
        log_text = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}.\d{3}', '', log_text)
        
        # 移除日志级别 (例如: "INFO", "ERROR")
        log_text = re.sub(r'\b(INFO|DEBUG|ERROR|WARNING|WARN|TRACE)\b', '', log_text)
        
        # 移除额外空格
        log_text = re.sub(r'\s+', ' ', log_text).strip()
        
        # 移除特殊字符并转为小写
        log_text = re.sub(r'[^a-zA-Z0-9\s]', '', log_text).lower()
        
        return log_text
    
    def tokenize(self, log_text):
        """将日志文本转换为token IDs"""
        processed_log = self.preprocess_log(log_text)
        tokens = self.tokenizer(processed_log, padding='max_length', truncation=True, 
                               max_length=128, return_tensors="pt")
        return tokens
    
    def text_to_token_list(self, log_text):
        """将日志文本转换为单词列表(用于人类可读)"""
        processed_log = self.preprocess_log(log_text)
        # 分割为单词列表
        words = processed_log.split()
        # 过滤掉空单词
        words = [word for word in words if word]
        return words

# 使用示例：
# tokenizer = LogTokenizer()
# log = "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"
# tokens = tokenizer.tokenize(log)
# word_list = tokenizer.text_to_token_list(log)
# print(word_list)  # ['dfsclient', 'successfully', 'read', 'block'] 