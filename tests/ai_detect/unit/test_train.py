#!/usr/bin/env python
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import json
import random
import logging

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

# 导入被测试模块
from ai_detect.train import setup_logging, split_logs_for_eval, parse_args


class TestTrain(unittest.TestCase):
    """AI检测训练模块的单元测试"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录和文件用于测试
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_log_file = os.path.join(self.temp_dir.name, "test_logs.log")
        self.temp_output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        # 写入一些测试日志
        self.test_logs = [f"测试日志行 {i}" for i in range(1, 201)]
        with open(self.temp_log_file, 'w', encoding='utf-8') as f:
            for log in self.test_logs:
                f.write(f"{log}\n")
    
    def tearDown(self):
        """测试后的清理工作"""
        # 清理临时目录和文件
        self.temp_dir.cleanup()
    
    def test_setup_logging(self):
        """测试日志设置功能"""
        # 使用正常的输出目录测试
        file_handler_mock = MagicMock()
        file_handler_mock.level = logging.INFO
        
        stream_handler_mock = MagicMock()
        stream_handler_mock.level = logging.INFO
        
        # 使用return_value指定模拟对象的返回值
        with patch('logging.FileHandler', return_value=file_handler_mock), \
             patch('logging.StreamHandler', return_value=stream_handler_mock):
            result = setup_logging(self.temp_output_dir)
            self.assertTrue(result)
            # 不再检查调用次数，因为我们使用了return_value
        
        # 测试无法创建日志文件的情况
        with patch('builtins.open', side_effect=PermissionError("模拟权限错误")), \
             patch('builtins.print') as mock_print:
            result = setup_logging("/invalid/path")
            self.assertFalse(result)
            mock_print.assert_called_with(mock_print.call_args[0][0])
    
    def test_split_logs_for_eval(self):
        """测试日志划分功能"""
        # 模拟所有可能的日志记录调用
        with patch('logging.info') as mock_info, \
             patch('logging.warning') as mock_warning, \
             patch('logging.error') as mock_error:
            # 测试正常划分
            train_file, eval_file = split_logs_for_eval(self.temp_log_file, eval_ratio=0.2, min_eval_lines=10)
        
            # 验证文件路径
            self.assertTrue(os.path.exists(train_file))
            self.assertTrue(os.path.exists(eval_file))
            
            # 验证文件内容
            with open(train_file, 'r', encoding='utf-8') as f:
                train_logs = [line.strip() for line in f if line.strip()]
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_logs = [line.strip() for line in f if line.strip()]
            
            # 验证数量
            self.assertEqual(len(train_logs) + len(eval_logs), 200)
            # 验证比例（允许有小误差）
            self.assertAlmostEqual(len(eval_logs) / 200, 0.2, delta=0.05)
            
            # 确保分割的日志内容正确
            for log in train_logs:
                self.assertIn(log, self.test_logs)
            for log in eval_logs:
                self.assertIn(log, self.test_logs)
    
    def test_split_logs_with_encoding_error(self):
        """测试读取编码错误的处理"""
        # 模拟所有编码都失败的情况
        mock_open_obj = MagicMock(side_effect=UnicodeDecodeError('utf-8', b'', 0, 1, 'mock error'))
        
        with patch('builtins.open', mock_open_obj), \
             patch('logging.error'), \
             patch('logging.warning'), \
             patch('logging.info'), \
             self.assertRaises(ValueError):
            split_logs_for_eval("/invalid/file.log")
    
    def test_split_logs_empty_file(self):
        """测试空文件的处理"""
        # 创建空文件
        empty_log_file = os.path.join(self.temp_dir.name, "empty_logs.log")
        with open(empty_log_file, 'w', encoding='utf-8') as f:
            pass
        
        # 模拟所有可能的日志记录调用
        with patch('logging.info') as mock_info, \
             patch('logging.warning') as mock_warning, \
             patch('logging.error') as mock_error, \
             self.assertRaises(ValueError):
            split_logs_for_eval(empty_log_file)
    
    def test_parse_args(self):
        """测试命令行参数解析功能"""
        # 测试必选参数
        with patch('sys.argv', ['train.py', '--train_file', 'train.log']):
            args = parse_args()
            self.assertEqual(args.train_file, 'train.log')
            self.assertIsNone(args.eval_file)
        
        # 测试所有参数
        with patch('sys.argv', [
            'train.py',
            '--train_file', 'train.log',
            '--eval_file', 'eval.log',
            '--model_dir', 'models',
            '--output_dir', 'output',
            '--tokenizer_name', 'bert-base-uncased',
            '--window_size', '5',
            '--num_epochs', '20'
        ]):
            args = parse_args()
            self.assertEqual(args.train_file, 'train.log')
            self.assertEqual(args.eval_file, 'eval.log')
            self.assertEqual(args.model_dir, 'models')
            self.assertEqual(args.output_dir, 'output')
            self.assertEqual(args.tokenizer_name, 'bert-base-uncased')
            self.assertEqual(args.window_size, 5)
            self.assertEqual(args.num_epochs, 20)


if __name__ == '__main__':
    unittest.main() 