from flask import Blueprint, request, jsonify,render_template
from app.services.nlp_processor import convert_to_sql
from app.models.db import get_db

nlp_bp = Blueprint("nlp", __name__)

@nlp_bp.route('/')
def index():
    return render_template('index.html')

@nlp_bp.route("/query", methods=["POST"])
def nlp_query():
    data = request.json
    user_query = data.get("query")

    # 解析自然语言查询
    sql_query = convert_to_sql(user_query)

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(sql_query)
    results = cursor.fetchall()

    return jsonify({"sql": sql_query, "results": [dict(row) for row in results]})