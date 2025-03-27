#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Flask应用入口文件
初始化并运行Flask应用
"""

import os
from flask import Flask, render_template
from flask_cors import CORS
from routes.nlp_routes import nlp_bp
from models.db import init_db

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 注册蓝图
app.register_blueprint(nlp_bp, url_prefix="/nlp")

# 初始化数据库
init_db()

@app.route("/")
def index():
    """首页路由"""
    return render_template("index.html")

if __name__ == "__main__":
    # 获取端口配置（默认5000）
    port = int(os.environ.get("PORT", 5000))
    # 运行应用
    app.run(host="0.0.0.0", port=port, debug=True) 