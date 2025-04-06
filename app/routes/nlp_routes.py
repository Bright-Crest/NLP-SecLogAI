#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NLP路由模块
处理自然语言查询请求，将自然语言转换为SQL查询
"""

from flask import Blueprint, request, jsonify
from services.nlp_processor import NL2SQLConverter
from models.db import get_db

nlp_bp = Blueprint("nlp", __name__)

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
    
    data = request.json
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"error": "缺少查询参数"}), 400
    
    # 使用NL2SQL转换器解析自然语言查询
    converter = NL2SQLConverter()
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
