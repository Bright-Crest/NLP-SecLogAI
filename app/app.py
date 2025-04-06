from flask import Flask, render_template
#from flask_migrate import Migrate
from app.models.db import  init_db
import logging

# ---------- 1. Flask应用初始化 ----------

app = Flask(__name__)

# 基础配置
app.config.update({
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///logs.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'JSONIFY_PRETTYPRINT_REGULAR': True  # 美化JSON输出
})


@app.route('/')
def index():
    return render_template('index.html')


# ---------- 2. 数据库初始化（兼容Flask-Migrate） ----------
_first_request_handled = False

@app.before_request
def initialize_database():
    global _first_request_handled
    if not _first_request_handled:
        init_db()  # 调用初始化函数
        _first_request_handled = True

# ---------- 3. 注册蓝图（按模块划分） ----------

from app.routes.main_routes import main_bp
from app.routes.log_routes import log_bp  # 成员B的日志管理API
from app.routes.nlp_routes import nlp_bp  # 成员D的NLP处理API
from app.routes.anomaly_routes import anomaly_bp  # 成员C的异常检测API
from app.routes.ai_routes import ai_bp, init_ai_bp  # 成员E的AI检测API

app.register_blueprint(main_bp, url_prefix='/')
app.register_blueprint(log_bp, url_prefix='/logs')
app.register_blueprint(nlp_bp, url_prefix='/nlp')
app.register_blueprint(anomaly_bp, url_prefix='/anomalies')
app.register_blueprint(ai_bp, url_prefix='/ai')

init_ai_bp(app)

# ---------- 4. 错误处理 ----------
# 注册错误处理
@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html'), 500

# ---------- 5. 日志记录配置 ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)



# ---------- 6. 启动应用 ----------
if __name__ == '__main__':

    # 初始化Flask-Migrate（需安装flask_migrate）
    # migrate = Migrate(app, get_db_connection)

    app.run(host='0.0.0.0', port=5000, debug=True)

