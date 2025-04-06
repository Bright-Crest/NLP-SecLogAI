#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP路由模块
处理自然语言查询请求，将自然语言转换为SQL查询
"""

import os
import sys
from flask import Blueprint, request, jsonify, render_template

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from services.nlp_processor import NL2SQLConverter
from models.db import get_db, TABLE_SCHEMA

nlp_bp = Blueprint("nlp", __name__)

converter = NL2SQLConverter(table_schema=TABLE_SCHEMA)


@nlp_bp.route("/query", methods=["POST"])
def nlp_query():
    """
    处理自然语言查询请求
    
    请求体:
    {
        "query": "最近24小时admin登录失败次数"
    }
    
    返回:
    {
        "sql": "SELECT COUNT(*) as failure_count FROM logs...",
        "results": [{"failure_count": 5}, ...],
        "original_query": "最近24小时admin登录失败次数"
    }
    """
    if not request.is_json:
        return jsonify({"error": "请求必须为JSON格式"}), 400
    
    data = request.json or {}
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"error": "缺少查询参数"}), 400
    
    # 使用NL2SQL转换器解析自然语言查询
    result = converter.convert(user_query)
    sql_query = result["sql"]
    
    # 执行SQL查询
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(sql_query)
        query_results = cursor.fetchall()
        
        # 将结果转换为JSON列表
        results = []
        for row in query_results:
            results.append(dict(row))
        
        return jsonify({
            "sql": sql_query,
            "results": results,
            "original_query": user_query
        })
    except Exception as e:
        return jsonify({
            "error": f"查询执行错误: {str(e)}",
            "sql": sql_query,
            "original_query": user_query
        }), 500 


@nlp_bp.route("/ui", methods=["GET"])
def nlp_ui():
    """
    自然语言转SQL的Web界面
    
    提供基于Bootstrap的前端界面，用于直观地进行自然语言查询
    """
    return render_template("nlp_sql.html")


# 添加CORS支持
@nlp_bp.after_request
def add_cors_headers(response):
    """为所有响应添加CORS头信息"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response 
