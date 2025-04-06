import unittest
import sys
import os
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.models.log_tokenizer import LogTokenizer

class TestLogTokenizer(unittest.TestCase):
    """测试LogTokenizer类的功能"""
    
    def setUp(self):
        """每个测试前的准备工作"""
        self.tokenizer = LogTokenizer()
        self.sample_log = "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"
        
    def test_preprocess_log(self):
        """测试日志预处理功能"""
        processed_log = self.tokenizer.preprocess_log(self.sample_log)
        # 检查时间戳已删除
        self.assertNotIn("2023-10-10", processed_log)
        # 检查日志级别已删除
        self.assertNotIn("INFO", processed_log)
        # 检查特殊字符已删除
        self.assertNotIn(":", processed_log)
        # 检查结果是小写的
        self.assertEqual(processed_log, processed_log.lower())
        # 检查输出包含关键字
        self.assertIn("dfsclient", processed_log)
        self.assertIn("successfully", processed_log)
        self.assertIn("read", processed_log)
        self.assertIn("block", processed_log)
        
    def test_text_to_token_list(self):
        """测试将文本转换为标记列表的功能"""
        token_list = self.tokenizer.text_to_token_list(self.sample_log)
        # 检查标记列表是否正确
        self.assertIsInstance(token_list, list)
        self.assertTrue(len(token_list) > 0)
        self.assertIn("dfsclient", token_list)
        self.assertIn("successfully", token_list)
        self.assertIn("read", token_list)
        self.assertIn("block", token_list)
        # 检查所有标记都是字符串
        for token in token_list:
            self.assertIsInstance(token, str)
            
    def test_tokenize(self):
        """测试文本分词功能"""
        tokens = self.tokenizer.tokenize(self.sample_log)
        # 检查返回对象是否含有必要的键
        self.assertIn("input_ids", tokens)
        self.assertIn("attention_mask", tokens)
        # 检查返回的是Pytorch张量
        self.assertIsInstance(tokens["input_ids"], torch.Tensor)
        self.assertIsInstance(tokens["attention_mask"], torch.Tensor)
        
        # 检查token大小
        self.assertEqual(tokens["input_ids"].shape[1], self.tokenizer.max_length)
        self.assertEqual(tokens["attention_mask"].shape[1], self.tokenizer.max_length)
        
    def test_tokenize_batch(self):
        """测试批量分词功能"""
        log_batch = [
            self.sample_log,
            "2023-10-11 09:15:45.678 ERROR Connection refused: connect",
            "2023-10-12 11:30:22.345 WARNING Low disk space on /dev/sda1"
        ]
        
        batch_tokens = self.tokenizer.tokenize_batch(log_batch)
        
        # 检查返回对象是否含有必要的键
        self.assertIn("input_ids", batch_tokens)
        self.assertIn("attention_mask", batch_tokens)
        # 检查返回的是Pytorch张量
        self.assertIsInstance(batch_tokens["input_ids"], torch.Tensor)
        self.assertIsInstance(batch_tokens["attention_mask"], torch.Tensor)
        
        # 检查批量大小
        self.assertEqual(batch_tokens["input_ids"].shape[0], len(log_batch))
        self.assertEqual(batch_tokens["attention_mask"].shape[0], len(log_batch))
        
        # 检查序列长度
        self.assertEqual(batch_tokens["input_ids"].shape[1], self.tokenizer.max_length)
        self.assertEqual(batch_tokens["attention_mask"].shape[1], self.tokenizer.max_length)
        
    def test_init_with_custom_tokenizer(self):
        """测试使用自定义分词器初始化"""
        custom_tokenizer = LogTokenizer(tokenizer_name="bert-base-uncased")
        # 确认初始化成功
        self.assertIsNotNone(custom_tokenizer.tokenizer)
        
        # 测试基本功能
        result = custom_tokenizer.preprocess_log(self.sample_log)
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main() 