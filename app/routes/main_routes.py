#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主路由模块
处理网站首页和一般页面的路由请求
"""

import os
import sys
from flask import Blueprint, render_template, redirect, url_for

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 创建蓝图
main_bp = Blueprint("main", __name__)


@main_bp.route('/', methods=['GET'])
def index():
    """
    网站首页
    
    提供了安全日志分析平台的主页面，介绍平台功能并导航到各个功能模块
    """
    return render_template('index.html')


# 简单的首页重定向
@main_bp.route('/index')
def index_redirect():
    return redirect('/')


@main_bp.route('/dashboard', methods=['GET'])
def dashboard():
    """
    仪表盘页面
    
    展示各项功能的综合数据和状态
    """
    # 这是一个占位功能，可以在未来实现
    return "仪表盘功能即将上线..."


@main_bp.route('/about', methods=['GET'])
def about():
    """
    关于页面
    
    展示项目介绍、团队成员、联系方式等信息
    """
    # 这是一个占位功能，可以在未来实现
    return render_template('index.html', _anchor='about')  # 跳转到首页的about部分


# 添加简单重定向，确保用户体验流畅
@main_bp.route('/ai', methods=['GET'])
def redirect_to_ai():
    """重定向到AI异常检测页面"""
    return redirect('/ai/ui')


@main_bp.route('/nlp', methods=['GET'])
def redirect_to_nlp():
    """重定向到自然语言查询页面"""
    return redirect('/nlp/ui') 