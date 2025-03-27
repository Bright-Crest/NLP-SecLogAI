#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
项目启动脚本
运行Flask应用
"""

import os
import sys

# 添加app目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# 导入app模块
from app import app

if __name__ == "__main__":
    # 获取端口号
    port = int(os.environ.get("PORT", 5000))
    # 运行应用
    app.run(host="0.0.0.0", port=port, debug=True) 