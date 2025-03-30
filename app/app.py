from flask import Flask
from app.routes.log_routes import log_bp
from app.routes.anomaly_routes import anomaly_bp
from app.routes.nlp_routes import nlp_bp
from app.models.db import init_db as db_init

app = Flask(__name__)

# 注册蓝图
app.register_blueprint(log_bp, url_prefix="/logs")
app.register_blueprint(anomaly_bp, url_prefix="/anomalies")
app.register_blueprint(nlp_bp, url_prefix="/nlp")

# 初始化数据库的函数
def init_db():
    db_init()

if __name__ == "__main__":
    app.run(debug=True)