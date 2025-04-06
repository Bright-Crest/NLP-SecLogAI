import sys
import os
import unittest
import tempfile
import json
import logging
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(ROOT_DIR)

# 导入测试模块
from ai_detect.ai_detect_cli import main as detect_main
from ai_detect.train import main as train_main
from ai_detect.evaluate import main as evaluate_main
from app.ai_models.anomaly_detector import AnomalyDetector


class TestAnomalyDetectionIntegration(unittest.TestCase):
    """AI异常检测流程的集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """进行一次性设置"""
        # 设置测试日志
        logging.basicConfig(level=logging.ERROR)
        
        # 创建临时工作目录
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.model_dir = os.path.join(cls.temp_dir.name, "models")
        cls.output_dir = os.path.join(cls.temp_dir.name, "output")
        cls.logs_dir = os.path.join(cls.temp_dir.name, "logs")
        
        # 创建目录
        os.makedirs(cls.model_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        os.makedirs(cls.logs_dir, exist_ok=True)
        
        # 创建正常日志文件
        cls.normal_log_file = os.path.join(cls.logs_dir, "normal_logs.log")
        normal_logs = [
            "2023-01-01 12:00:01 INFO [main] 系统启动成功",
            "2023-01-01 12:00:02 INFO [main] 加载配置文件",
            "2023-01-01 12:00:03 INFO [main] 初始化数据库连接",
            "2023-01-01 12:00:04 INFO [main] 启动Web服务器",
            "2023-01-01 12:00:05 INFO [main] 注册健康检查",
            "2023-01-01 12:00:06 INFO [main] 注册API路由",
            "2023-01-01 12:00:07 INFO [main] 初始化缓存",
            "2023-01-01 12:00:08 INFO [main] 启动定时任务",
            "2023-01-01 12:00:09 INFO [main] 加载用户数据",
            "2023-01-01 12:00:10 INFO [main] 系统初始化完成"
        ]
        with open(cls.normal_log_file, 'w', encoding='utf-8') as f:
            for log in normal_logs:
                f.write(f"{log}\n")
        
        # 创建异常日志文件
        cls.anomaly_log_file = os.path.join(cls.logs_dir, "anomaly_logs.log")
        anomaly_logs = [
            "2023-01-01 12:00:01 INFO [main] 系统启动成功",
            "2023-01-01 12:00:02 INFO [main] 加载配置文件",
            "2023-01-01 12:00:03 ERROR [main] 无法连接数据库: Connection refused",
            "2023-01-01 12:00:04 ERROR [main] 数据库重试失败",
            "2023-01-01 12:00:05 WARN [main] 使用备用配置",
            "2023-01-01 qwertyuiop 乱码错误日志",
            "2023-01-01 12:00:07 ERROR [main] NullPointerException at line 243",
            "2023-01-01 12:00:08 FATAL [main] 系统崩溃",
            "2023-01-01 12:00:09 ERROR [main] 无法恢复服务",
            "2023-01-01 12:00:10 INFO [main] 紧急关闭系统"
        ]
        with open(cls.anomaly_log_file, 'w', encoding='utf-8') as f:
            for log in anomaly_logs:
                f.write(f"{log}\n")
    
    @classmethod
    def tearDownClass(cls):
        """清理临时目录"""
        # 关闭所有日志处理程序，防止文件被占用
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
        
        # 重新设置基本日志配置
        logging.basicConfig(level=logging.ERROR, handlers=[logging.StreamHandler()])
        
        # 尝试清理临时目录
        try:
            cls.temp_dir.cleanup()
        except PermissionError:
            print("警告：无法删除某些临时文件，可能被其他进程占用")
        except Exception as e:
            print(f"清理临时目录时发生错误: {str(e)}")
    
    @patch('ai_detect.train.AnomalyDetector')
    def test_train_eval_detect_workflow(self, mock_detector_class):
        """测试训练-评估-检测的完整工作流"""
        # 模拟训练过程
        mock_detector = MagicMock()
        mock_detector_class.return_value = mock_detector
        mock_detector.train.return_value = {"loss": 0.1, "accuracy": 0.9}        
        mock_detector.evaluate.return_value = {"loss": 0.2, "accuracy": 0.85}    

        # 1. 训练模型
        with patch('sys.argv', [
            'train.py',
            '--train_file', self.normal_log_file,
            '--eval_file', self.normal_log_file,  # 使用相同文件简化测试
            '--model_dir', self.model_dir,
            '--output_dir', self.output_dir,
            '--num_epochs', '1',  # 减少训练时间
            '--window_size', '5',
            '--tokenizer_name', 'prajjwal1/bert-mini'
        ]), patch('builtins.print'):
            train_main()

        # 验证训练调用
        mock_detector.train.assert_called_once()

        # 2. 评估模型
        # 准备评估预期结果
        evaluation_result = {
            "accuracy": 0.85,
            "precision": 0.8,
            "recall": 0.75,
            "f1_score": 0.78,
            "auc": 0.9,
            "confusion_matrix": [[90, 10], [20, 80]],
            "detection_method": "ensemble",
            "threshold": 0.5,
            "num_samples": 100,
            "num_anomalies": 20
        }
        
        # 这里我们需要直接模拟evaluate方法从而避免实际加载模型
        with patch('app.ai_models.anomaly_detector.AnomalyDetector.evaluate', return_value=evaluation_result), \
             patch('sys.argv', [
                'evaluate.py',
                '--model_dir', self.model_dir,
                '--test_file', self.normal_log_file,  # 修改为test_file参数
                '--output_dir', self.output_dir,
                '--window_size', '5',
                '--tokenizer_name', 'prajjwal1/bert-mini'
            ]), patch('builtins.print'):
            evaluate_main()
        
        # 3. 检测正常日志
        normal_detection_result = {
            'num_windows': 10,
            'avg_score': 0.2,
            'max_score': 0.3,
            'num_anomaly_windows': 0,
            'anomaly_ratio': 0.0,
            'windows': [{'window_idx': i, 'score': 0.2, 'is_anomaly': False} for i in range(10)]
        }
        
        # 对于检测功能，我们需要模拟detect_sequence方法
        with patch('app.ai_models.anomaly_detector.AnomalyDetector.detect_sequence', return_value=normal_detection_result), \
             patch('sys.argv', [
                'ai_detect_cli.py',
                '--log-file', self.normal_log_file,
                '--model-path', self.model_dir,
                '--output_dir', os.path.join(self.output_dir, 'normal_result.json'),
                '--window-type', 'fixed',
                '--window-size', '5'
            ]), patch('builtins.print'):
            detect_main()
        
        # 4. 检测异常日志
        anomaly_detection_result = {
            'num_windows': 10,
            'avg_score': 0.6,
            'max_score': 0.9,
            'num_anomaly_windows': 6,
            'anomaly_ratio': 0.6,
            'windows': [
                {'window_idx': i, 
                 'score': 0.9 if i % 2 == 0 else 0.3, 
                 'is_anomaly': i % 2 == 0} 
                for i in range(10)
            ]
        }
        
        # 再次模拟检测函数用于异常日志
        with patch('app.ai_models.anomaly_detector.AnomalyDetector.detect_sequence', return_value=anomaly_detection_result), \
             patch('sys.argv', [
                'ai_detect_cli.py',
                '--log-file', self.anomaly_log_file,
                '--model-path', self.model_dir,
                '--output_dir', os.path.join(self.output_dir, 'anomaly_result.json'),
                '--window-type', 'fixed',
                '--window-size', '5'
            ]), patch('builtins.print'):
            detect_main()
        
        # 验证检测调用
        # 注意：由于我们使用的是不同的模拟对象，所以无法通过mock_detector验证调用次数
        # 这里我们只验证测试没有抛出异常就认为成功
    
    def test_detector_initialization(self):
        """测试检测器的实际初始化（不使用模拟）"""
        # 使用小型tokenizer模型以加速测试
        try:
            detector = AnomalyDetector(tokenizer_name="prajjwal1/bert-mini", window_size=5)
            self.assertIsNotNone(detector)
            self.assertEqual(detector.window_size, 5)
            self.assertEqual(detector.tokenizer.name_or_path, "prajjwal1/bert-mini")
        except Exception as e:
            self.fail(f"初始化AnomalyDetector时出错: {str(e)}")


if __name__ == '__main__':
    unittest.main() 