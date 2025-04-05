# 项目核心配置
import os

class config:
    # Flask基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True  # 开发模式开启
    
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = 'sqlite:///logs.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 异常检测规则配置
    BRUTE_FORCE_THRESHOLD = 10  # 5分钟内最大失败尝试次数
    ABNORMAL_IP_THRESHOLD = 3   # 24小时内最大不同IP数
    
    # NLP配置
    NLP_MODEL_PATH = "models/nlp_model"
    ENABLE_GPT = False  # 是否启用GPT-4

class ProductionConfig(config):
    DEBUG = False

class DevelopmentConfig(config):
    pass

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}