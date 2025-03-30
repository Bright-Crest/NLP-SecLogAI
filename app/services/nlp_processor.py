import re

def convert_to_sql(nlp_query):
    """
    解析自然语言查询，转换为 SQL 语句
    """
    if "最近 24 小时" in nlp_query:
        return "SELECT * FROM logs WHERE timestamp >= datetime('now', '-1 day')"

    if "admin 登录失败" in nlp_query:
        return "SELECT * FROM logs WHERE username='admin' AND event='login_failed'"

    return "SELECT * FROM logs"