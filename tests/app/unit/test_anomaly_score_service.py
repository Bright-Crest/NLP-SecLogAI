import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from app.services.anomaly_score import AnomalyScoreService

class TestAnomalyScoreService(unittest.TestCase):
    """测试AnomalyScoreService类的功能"""
    
    def setUp(self):
        """每个测试前的准备工作，使用Mock替代实际模型"""
        # 使用patch替代实际的AnomalyDetector以避免加载真实模型
        self.detector_patch = patch('app.services.anomaly_score.AnomalyDetector')
        self.mock_detector = self.detector_patch.start()
        
        # 配置mock对象
        self.mock_detector_instance = MagicMock()
        self.mock_detector.return_value = self.mock_detector_instance
        
        # 使用patch替代LogWindow
        self.log_window_patch = patch('app.services.anomaly_score.LogWindow')
        self.mock_log_window = self.log_window_patch.start()
        self.mock_log_window_instance = MagicMock()
        self.mock_log_window.return_value = self.mock_log_window_instance
        
        # 创建服务实例
        self.service = AnomalyScoreService(
            model_dir=None,
            window_size=5,
            tokenizer_name='bert-mini',
            detection_method='ensemble'
        )
        
        # 设置测试用例
        self.sample_log = "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"
        self.sample_logs = [
            self.sample_log,
            "2023-10-11 09:15:45.678 ERROR Connection refused: connect",
            "2023-10-12 11:30:22.345 WARNING Low disk space on /dev/sda1"
        ]
    
    def tearDown(self):
        """每个测试后的清理工作"""
        self.detector_patch.stop()
        self.log_window_patch.stop()
    
    def test_init(self):
        """测试初始化"""
        # 检查初始化正确设置了参数
        self.assertEqual(self.service.window_size, 5)
        self.assertEqual(self.service.batch_size, 128)  # 默认值
        
        # 检查是否正确调用了AnomalyDetector
        self.mock_detector.assert_called_once_with(
            model_dir=None,
            window_size=5,
            tokenizer_name='bert-mini',
            detection_method='ensemble'
        )
    
    def test_score_single_log_without_knn(self):
        """测试单条日志评分（不使用KNN）"""
        # 设置mock返回值
        expected_score = 0.75
        self.mock_detector_instance.detect.return_value = {
            'score': expected_score,
            'is_anomaly': True
        }
        
        # 调用待测试方法
        score = self.service.score_single_log(self.sample_log, use_knn=False)
        
        # 验证结果
        self.assertEqual(score, expected_score)
        self.mock_detector_instance.detect.assert_called_once_with(
            log_text=self.sample_log,
            threshold=self.service.threshold
        )
    
    def test_score_log_sequence_without_knn(self):
        """测试日志序列评分（不使用KNN）"""
        # 设置mock返回值
        window_scores = [0.3, 0.8, 0.4]
        avg_score = 0.5
        max_score = 0.8
        
        self.mock_detector_instance.detect_sequence.return_value = {
            'windows': [
                {'score': window_scores[0]}, 
                {'score': window_scores[1]}, 
                {'score': window_scores[2]}
            ],
            'avg_score': avg_score,
            'max_score': max_score
        }
        
        # 调用待测试方法
        scores, avg, max_val = self.service.score_log_sequence(
            self.sample_logs, 
            window_type='fixed', 
            stride=1, 
            use_knn=False
        )
        
        # 验证结果
        self.assertEqual(scores, window_scores)
        self.assertEqual(avg, avg_score)
        self.assertEqual(max_val, max_score)
        
        # 验证方法调用
        self.mock_detector_instance.detect_sequence.assert_called_once_with(
            log_list=self.sample_logs,
            window_type='fixed',
            stride=1,
            threshold=self.service.threshold,
            batch_size=self.service.batch_size
        )
    
    def test_is_anomaly(self):
        """测试异常判断功能"""
        # 设置阈值
        self.service.threshold = 0.6
        
        # 测试低于阈值（正常）
        self.assertFalse(self.service.is_anomaly(0.5))
        
        # 测试高于阈值（异常）
        self.assertTrue(self.service.is_anomaly(0.7))
        
        # 测试等于阈值（正常）
        self.assertFalse(self.service.is_anomaly(0.6))
    
    def test_set_threshold(self):
        """测试设置阈值功能"""
        # 初始值
        initial_threshold = self.service.threshold
        
        # 设置新阈值
        new_threshold = 0.75
        self.service.set_threshold(new_threshold)
        
        # 验证是否更新
        self.assertEqual(self.service.threshold, new_threshold)
        self.assertNotEqual(self.service.threshold, initial_threshold)
    
    def test_set_use_knn(self):
        """测试设置KNN使用状态功能"""
        # 初始值
        initial_use_knn = self.service.use_knn
        
        # 设置新状态
        self.service.set_use_knn(not initial_use_knn)
        
        # 验证是否更新
        self.assertEqual(self.service.use_knn, not initial_use_knn)


if __name__ == "__main__":
    unittest.main() 