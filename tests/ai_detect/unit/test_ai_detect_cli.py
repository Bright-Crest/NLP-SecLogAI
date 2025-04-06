#!/usr/bin/env python
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import json

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

# 导入被测试模块
from ai_detect.ai_detect_cli import setup_logging, parse_args, main


class TestAiDetectCli(unittest.TestCase):
    """AI检测CLI模块的单元测试"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建临时目录和文件用于测试
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_log_file = os.path.join(self.temp_dir.name, "test_logs.log")
        self.temp_output_dir = os.path.join(self.temp_dir.name, "output")
        
        # 写入一些测试日志
        with open(self.temp_log_file, 'w', encoding='utf-8') as f:
            f.write("这是测试日志1\n")
            f.write("这是测试日志2\n")
            f.write("这是测试日志3\n")
    
    def tearDown(self):
        """测试后的清理工作"""
        # 清理临时目录和文件
        self.temp_dir.cleanup()
    
    def test_setup_logging(self):
        """测试日志设置功能"""
        # 测试不提供输出目录的情况
        with patch('logging.StreamHandler') as mock_handler:
            setup_logging()
            mock_handler.assert_called_once()
        
        # 测试提供输出目录的情况
        with patch('logging.StreamHandler') as mock_stream_handler, \
             patch('logging.FileHandler') as mock_file_handler, \
             patch('os.makedirs') as mock_makedirs:
            setup_logging(self.temp_output_dir)
            mock_makedirs.assert_called_once_with(self.temp_output_dir, exist_ok=True)
            mock_stream_handler.assert_called_once()
            mock_file_handler.assert_called_once()
    
    def test_parse_args(self):
        """测试命令行参数解析功能"""
        # 测试使用log_file参数
        with patch('sys.argv', ['ai_detect_cli.py', '--log-file', self.temp_log_file]):
            args = parse_args()
            self.assertEqual(args.log_file, self.temp_log_file)
            self.assertIsNone(args.logs)
            
        # 测试使用logs参数
        with patch('sys.argv', ['ai_detect_cli.py', '--logs', '日志1', '日志2']):
            args = parse_args()
            self.assertEqual(args.logs, ['日志1', '日志2'])
            self.assertIsNone(args.log_file)
        
        # 测试其他参数的默认值
        with patch('sys.argv', ['ai_detect_cli.py', '--logs', '日志1']):
            args = parse_args()
            self.assertEqual(args.window_type, 'fixed')
            self.assertEqual(args.window_size, 10)
            self.assertEqual(args.stride, 1)
            self.assertEqual(args.threshold, 0.5)
    
    @patch('ai_detect.ai_detect_cli.AnomalyDetector')
    def test_main_single_log(self, mock_detector_class):
        """测试单条日志检测主函数"""
        # 设置模拟对象
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect.return_value = {
            'log': '这是测试日志1', 
            'score': 0.3, 
            'threshold': 0.5, 
            'is_anomaly': False
        }
        
        # 运行测试：单条日志检测
        with patch('sys.argv', ['ai_detect_cli.py', '--logs', '这是测试日志1']), \
             patch('builtins.print') as mock_print:
            main()
            
            # 验证
            mock_detector_class.assert_called_once()
            mock_detector.detect.assert_called_once()
            mock_print.assert_any_call("\n=== 检测结果 ===")
            mock_print.assert_any_call("日志: 这是测试日志1")
            mock_print.assert_any_call("异常分数: 0.3000")
            mock_print.assert_any_call("阈值: 0.5")
            mock_print.assert_any_call("结论: 【正常】")
    
    @patch('ai_detect.ai_detect_cli.AnomalyDetector')
    def test_main_multiple_logs(self, mock_detector_class):
        """测试多条日志检测主函数"""
        # 设置模拟对象
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_sequence.return_value = {
            'num_windows': 3,
            'avg_score': 0.4,
            'max_score': 0.6,
            'num_anomaly_windows': 1,
            'anomaly_ratio': 0.333,
            'windows': [
                {'window_idx': 0, 'score': 0.3, 'is_anomaly': False},
                {'window_idx': 1, 'score': 0.6, 'is_anomaly': True},
                {'window_idx': 2, 'score': 0.3, 'is_anomaly': False}
            ]
        }
        
        # 运行测试：使用日志文件
        with patch('sys.argv', ['ai_detect_cli.py', '--log-file', self.temp_log_file]), \
             patch('builtins.print') as mock_print:
            main()
            
            # 验证
            mock_detector_class.assert_called_once()
            mock_detector.detect_sequence.assert_called_once()
            mock_print.assert_any_call("\n=== 检测结果摘要 ===")
            mock_print.assert_any_call("窗口数量: 3")
            mock_print.assert_any_call("平均异常分数: 0.4000")
            mock_print.assert_any_call("最大异常分数: 0.6000")
            mock_print.assert_any_call("异常窗口数量: 1")
            mock_print.assert_any_call("异常比例: 33.30%")
    
    @patch('ai_detect.ai_detect_cli.AnomalyDetector')
    def test_main_with_errors(self, mock_detector_class):
        """测试主函数中的错误处理"""
        # 测试模型路径不存在的情况
        with patch('sys.argv', ['ai_detect_cli.py', '--logs', '测试日志']), \
             patch('os.path.exists', return_value=False), \
             patch('logging.error') as mock_logging_error:
            main()
            mock_logging_error.assert_called_with(mock_logging_error.call_args[0][0])
        
        # 测试检测器初始化失败的情况
        mock_detector_class.side_effect = Exception("模拟的初始化错误")
        with patch('sys.argv', ['ai_detect_cli.py', '--logs', '测试日志']), \
             patch('os.path.exists', return_value=True), \
             patch('logging.error') as mock_logging_error:
            main()
            mock_logging_error.assert_called_with("初始化检测器失败: 模拟的初始化错误")


if __name__ == '__main__':
    unittest.main() 