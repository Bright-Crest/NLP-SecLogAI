import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 为werkzeug添加__version__属性以修复测试客户端问题
import werkzeug
if not hasattr(werkzeug, '__version__'):
    werkzeug.__version__ = '2.3.0'  # 使用一个合理的版本号

# 导入flask相关模块
import flask
from flask import Flask
from app.routes.ai_routes import ai_bp

# 测试类定义
class TestAIRoutes(unittest.TestCase):
    """测试AI路由的API端点"""
    
    def setUp(self):
        """每个测试前的准备工作"""
        # 创建Flask应用
        self.app = Flask(__name__)
        # 注册蓝图
        self.app.register_blueprint(ai_bp)
        
        # 修改创建测试客户端的方式，避免使用werkzeug.__version__
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # 设置测试数据
        self.sample_log = "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"
        self.sample_logs = [
            self.sample_log,
            "2023-10-11 09:15:45.678 ERROR Connection refused: connect",
            "2023-10-12 11:30:22.345 WARNING Low disk space on /dev/sda1"
        ]
        
        # 模拟anomaly_service
        self.mock_service = MagicMock()
        self.mock_service.threshold = 0.5
        self.mock_service.use_knn = False
        self.mock_service.knn_model = None
        
        # 当测试中调用get_anomaly_service时，会直接返回我们的mock_service
        # 这样就不需要实际调用init_ai_services来初始化服务
        with patch('app.routes.ai_routes.get_anomaly_service', return_value=self.mock_service):
            pass  # 这里只是为了确保服务已经被模拟，不需要实际执行
    
    @patch('app.routes.ai_routes.get_anomaly_service')
    def test_score_log_endpoint(self, mock_get_service):
        """测试单条日志评分端点"""
        # 配置mock
        mock_get_service.return_value = self.mock_service
        self.mock_service.score_single_log.return_value = 0.75
        self.mock_service.is_anomaly.return_value = True
        
        # 发送请求
        response = self.client.post(
            '/ai/score_log',
            data=json.dumps({
                'log': self.sample_log
            }),
            content_type='application/json'
        )
        
        # 断言响应
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('score', data)
        self.assertIn('is_anomaly', data)
        self.assertEqual(data['score'], 0.75)
        self.assertEqual(data['is_anomaly'], True)
        
        # 验证服务调用
        self.mock_service.score_single_log.assert_called_once_with(
            self.sample_log, use_knn=None
        )
    
    @patch('app.routes.ai_routes.get_anomaly_service')
    def test_score_log_sequence_endpoint(self, mock_get_service):
        """测试日志序列评分端点"""
        # 配置mock
        mock_get_service.return_value = self.mock_service
        
        # 模拟评分结果
        scores = [0.3, 0.8, 0.4]
        avg_score = 0.5
        max_score = 0.8
        self.mock_service.score_log_sequence.return_value = (scores, avg_score, max_score)
        
        # 发送请求
        response = self.client.post(
            '/ai/score_log_sequence',
            data=json.dumps({
                'logs': self.sample_logs,
                'window_type': 'fixed',
                'stride': 1
            }),
            content_type='application/json'
        )
        
        # 断言响应
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('scores', data)
        self.assertIn('avg_score', data)
        self.assertIn('max_score', data)
        self.assertEqual(data['scores'], scores)
        self.assertEqual(data['avg_score'], avg_score)
        self.assertEqual(data['max_score'], max_score)
        self.assertEqual(data['anomaly_windows'], [1])  # 分数0.8超过阈值0.5
        
        # 验证服务调用
        self.mock_service.score_log_sequence.assert_called_once_with(
            self.sample_logs, window_type='fixed', stride=1, use_knn=None
        )
    
    @patch('app.routes.ai_routes.get_anomaly_service')
    def test_detect_anomaly_endpoint(self, mock_get_service):
        """测试异常检测端点"""
        # 配置mock
        mock_get_service.return_value = self.mock_service
        
        # 模拟评分结果
        scores = [0.3, 0.8, 0.4]
        avg_score = 0.5
        max_score = 0.8
        self.mock_service.score_log_sequence.return_value = (scores, avg_score, max_score)
        
        # 模拟异常检测结果
        result = {
            'num_windows': 3,
            'avg_score': 0.5,
            'max_score': 0.8,
            'num_anomaly_windows': 1,
            'anomaly_ratio': 0.33,
            'windows': [
                {'score': 0.3, 'is_anomaly': False},
                {'score': 0.8, 'is_anomaly': True},
                {'score': 0.4, 'is_anomaly': False}
            ]
        }
        self.mock_service.detector.detect_sequence.return_value = result
        
        # 自定义window_size以便窗口计算
        self.mock_service.window_size = len(self.sample_logs)

        # 发送请求
        response = self.client.post(
            '/ai/detect',
            data=json.dumps({
                'logs': self.sample_logs,
                'window_type': 'sliding',
                'stride': 1,
                'threshold': 0.6
            }),
            content_type='application/json'
        )

        # 断言响应
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('result', data)
        
        # 检查关键字段是否匹配
        self.assertEqual(data['result']['avg_score'], result['avg_score'])
        self.assertEqual(data['result']['max_score'], result['max_score'])
        self.assertEqual(data['result']['num_windows'], result['num_windows'])
    
    @patch('app.routes.ai_routes.get_anomaly_service')
    def test_model_threshold_endpoint(self, mock_get_service):
        """测试设置模型阈值端点"""
        # 配置mock
        mock_get_service.return_value = self.mock_service
        
        # 发送请求
        response = self.client.post(
            '/ai/model/threshold',
            data=json.dumps({
                'threshold': 0.75
            }),
            content_type='application/json'
        )
        
        # 断言响应
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        
        # 验证服务调用
        self.mock_service.set_threshold.assert_called_once_with(0.75)
    
    @patch('app.routes.ai_routes.get_anomaly_service')
    def test_knn_status_endpoint(self, mock_get_service):
        """测试KNN状态端点"""
        # 配置mock
        mock_get_service.return_value = self.mock_service
        self.mock_service.use_knn = True
        
        # 发送请求 - 获取状态
        response = self.client.get('/ai/knn/status')
        
        # 断言响应
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('knn_enabled', data)
        self.assertEqual(data['knn_enabled'], True)
        
        # 发送请求 - 设置状态
        response = self.client.post(
            '/ai/knn/status',
            data=json.dumps({
                'enabled': False
            }),
            content_type='application/json'
        )
        
        # 断言响应
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        
        # 验证服务调用
        self.mock_service.set_use_knn.assert_called_once_with(False)


if __name__ == "__main__":
    unittest.main() 