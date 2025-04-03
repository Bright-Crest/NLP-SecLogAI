from flask import Blueprint, request, jsonify
from app.services.anomaly_score import get_anomaly_score, is_log_anomaly
from app.services.log_tokenizer import LogTokenizer
import time

# 创建蓝图
anomaly_bp = Blueprint('anomaly', __name__, url_prefix='/api/anomaly')

# 初始化分词器
tokenizer = LogTokenizer()

@anomaly_bp.route('/detect', methods=['POST'])
def detect_anomaly():
    """
    接收日志文本，返回异常检测结果
    
    请求格式:
        {
            "log_text": "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx",
            "threshold": 0.6  # 可选参数，自定义阈值
        }
    
    返回格式:
        {
            "is_anomaly": true/false,
            "anomaly_score": 0.85,
            "tokens": ["dfsclient", "successfully", "read", "block"],
            "processing_time_ms": 125
        }
    """
    # 获取请求数据
    data = request.json
    if not data or 'log_text' not in data:
        return jsonify({"error": "请求数据缺少'log_text'字段"}), 400
    
    log_text = data.get('log_text')
    threshold = data.get('threshold')  # 可选参数
    
    start_time = time.time()
    
    # 转换为token列表（便于前端理解）
    tokens = tokenizer.text_to_token_list(log_text)
    
    # 检测异常
    is_anomaly, score = is_log_anomaly(log_text, threshold)
    
    # 计算处理时间
    processing_time = (time.time() - start_time) * 1000  # 毫秒
    
    # 返回结果
    result = {
        "is_anomaly": is_anomaly,
        "anomaly_score": score,
        "tokens": tokens,
        "processing_time_ms": processing_time
    }
    
    return jsonify(result)

@anomaly_bp.route('/batch_detect', methods=['POST'])
def batch_detect_anomaly():
    """
    批量检测日志
    
    请求格式:
        {
            "logs": [
                "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx",
                "2023-10-10 08:02:31.456 ERROR DFSClient: Failed to read block BP-yyy"
            ],
            "threshold": 0.6  # 可选参数，自定义阈值
        }
    
    返回格式:
        {
            "results": [
                {
                    "log_text": "...",
                    "is_anomaly": false,
                    "anomaly_score": 0.25,
                    "tokens": [...] 
                },
                {...}
            ],
            "summary": {
                "total_logs": 2,
                "anomaly_count": 1,
                "anomaly_percentage": 50.0,
                "processing_time_ms": 250
            }
        }
    """
    # 获取请求数据
    data = request.json
    if not data or 'logs' not in data:
        return jsonify({"error": "请求数据缺少'logs'字段"}), 400
    
    logs = data.get('logs')
    if not isinstance(logs, list):
        return jsonify({"error": "'logs'必须是日志文本列表"}), 400
    
    threshold = data.get('threshold')  # 可选参数
    
    start_time = time.time()
    results = []
    anomaly_count = 0
    
    # 处理每条日志
    for log_text in logs:
        # 检测异常
        is_anomaly, score = is_log_anomaly(log_text, threshold)
        if is_anomaly:
            anomaly_count += 1
        
        # 转换为token列表
        tokens = tokenizer.text_to_token_list(log_text)
        
        # 添加到结果列表
        results.append({
            "log_text": log_text,
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "tokens": tokens
        })
    
    # 计算处理时间
    processing_time = (time.time() - start_time) * 1000  # 毫秒
    
    # 计算统计数据
    total_logs = len(logs)
    anomaly_percentage = (anomaly_count / total_logs * 100) if total_logs > 0 else 0
    
    # 返回结果
    response = {
        "results": results,
        "summary": {
            "total_logs": total_logs,
            "anomaly_count": anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "processing_time_ms": processing_time
        }
    }
    
    return jsonify(response)

@anomaly_bp.route('/tokenize', methods=['POST'])
def tokenize_log():
    """
    将日志文本转换为token列表
    
    请求格式:
        {
            "log_text": "2023-10-10 08:02:30.123 INFO DFSClient: Successfully read block BP-xxx"
        }
    
    返回格式:
        {
            "tokens": ["dfsclient", "successfully", "read", "block"]
        }
    """
    # 获取请求数据
    data = request.json
    if not data or 'log_text' not in data:
        return jsonify({"error": "请求数据缺少'log_text'字段"}), 400
    
    log_text = data.get('log_text')
    
    # 转换为token列表
    tokens = tokenizer.text_to_token_list(log_text)
    
    return jsonify({"tokens": tokens})

# 为健康检查提供一个简单的端点
@anomaly_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "service": "anomaly_detection"}) 