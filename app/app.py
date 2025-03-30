from flask import Flask
from flask_migrate import Migrate
from .models.db import get_db_connection, init_db
import logging


# ---------- 1. Flask应用初始化 ----------
def create_app():
    app = Flask(__name__)

    # 基础配置
    app.config.update({
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///security_logs.db',
        'SQLALCHEMY_TRACK_MODIFICATIONS': False,
        'JSONIFY_PRETTYPRINT_REGULAR': True  # 美化JSON输出
    })

    # ---------- 2. 数据库初始化（兼容Flask-Migrate） ----------
    _first_request_handled = False

    @app.before_request
    def initialize_database():
        global _first_request_handled
        if not _first_request_handled:
            init_db()  # 调用初始化函数
            _first_request_handled = True

    # ---------- 3. 注册蓝图（按模块划分） ----------
    from .routes.log_routes import log_bp  # 成员B的日志管理API
    from .routes.rule_routes import rule_bp  # 成员C的规则检测API
    from .routes.nlp_routes import nlp_bp  # 成员D的NLP处理API
    from .routes.anomaly_routes import anomaly_bp  # 成员C的异常检测API

    app.register_blueprint(log_bp, url_prefix='/logs')
    app.register_blueprint(rule_bp, url_prefix='/rules')
    app.register_blueprint(nlp_bp, url_prefix='/nlp')
    app.register_blueprint(anomaly_bp, url_prefix='/anomalies')

    # ---------- 4. 错误处理 ----------
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'API endpoint not found'}, 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Server Error: {error}")
        return {'error': 'Internal server error'}, 500

    # ---------- 5. 日志记录配置 ----------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )

    return app


# ---------- 6. 启动应用 ----------
if __name__ == '__main__':
    app = create_app()

    # 初始化Flask-Migrate（需安装flask_migrate）
    migrate = Migrate(app, get_db_connection)

    app.run(host='0.0.0.0', port=5000, debug=True)