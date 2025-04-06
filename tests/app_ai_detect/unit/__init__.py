"""
APP模块单元测试包
包含对app模块主要功能的单元测试
"""

import os
import sys

# 将项目根目录添加到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# 导出测试类
from tests.app_ai_detect.unit.test_log_tokenizer import TestLogTokenizer
from tests.app_ai_detect.unit.test_log_window import TestLogWindow
from tests.app_ai_detect.unit.test_anomaly_score_service import TestAnomalyScoreService
from tests.app_ai_detect.unit.test_ai_routes import TestAIRoutes

# 记录版本信息
__version__ = '0.1.0'