# 项目核心配置
import os

class config:
    # Flask基础配置
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True  # 开发模式开启
    
    # 日志数据库配置
    SQLALCHEMY_DATABASE_URI = 'sqlite:///logs.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    

class ProductionConfig(config):
    DEBUG = False

class DevelopmentConfig(config):
    pass

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}