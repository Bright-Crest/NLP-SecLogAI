from flask import Blueprint, request, jsonify
import os
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.anomaly_score import get_anomaly_service, init_anomaly_service
from app.models.anomaly_detector import AnomalyDetector

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT_DIR, 'ai_detect', 'checkpoint')

# 创建蓝图
ai_bp = Blueprint('ai', __name__, url_prefix='/ai')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局检测器
detector = None

@ai_bp.before_app_first_request
def setup_ai_services():
    """初始化AI检测服务"""
    global detector
    
    # 默认模型路径
    model_dir = os.environ.get('AI_MODEL_PATH', MODEL_DIR)
    
    # 检查模型是否存在
    if os.path.exists(model_dir):
        # 初始化异常评分服务
        init_anomaly_service(model_dir=model_dir)
        logger.info(f"异常评分服务已初始化，使用模型: {model_dir}")
        
        # 初始化检测器
        detector = AnomalyDetector()
        logger.info("AI检测器已初始化")
    else:
        logger.warning(f"模型文件不存在: {model_dir}")


@ai_bp.route('/score_log', methods=['POST'])
def score_log():
    """
    对单条日志进行异常评分
    
    请求体:
    {
        "log": "日志文本"
    }
    
    响应:
    {
        "score": 0.82,
        "is_anomaly": true,
        "threshold": 0.5
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'log' not in data:
            return jsonify({"error": "请求体必须包含'log'字段"}), 400
        
        log_text = data['log']
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 计算异常分数
        score = anomaly_service.score_single_log(log_text)
        
        # 判断是否异常
        threshold = data.get('threshold', anomaly_service.threshold)
        is_anomaly = anomaly_service.is_anomaly(score)
        
        # 返回结果
        return jsonify({
            "log": log_text,
            "score": float(score),
            "threshold": float(threshold),
            "is_anomaly": bool(is_anomaly)
        })
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@ai_bp.route('/score_log_sequence', methods=['POST'])
def score_log_sequence():
    """
    对日志序列进行异常评分
    
    请求体:
    {
        "logs": ["日志1", "日志2", ...],
        "window_type": "fixed",  # 可选，默认为"fixed"
        "stride": 1              # 可选，默认为1
    }
    
    响应:
    {
        "scores": [0.1, 0.2, ...],  # 每个窗口的分数
        "avg_score": 0.15,
        "max_score": 0.2,
        "anomaly_windows": [...]     # 异常窗口索引
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'logs' not in data:
            return jsonify({"error": "请求体必须包含'logs'字段"}), 400
        
        logs = data['logs']
        window_type = data.get('window_type', 'fixed')
        stride = data.get('stride', 1)
        
        # 验证窗口类型
        if window_type not in ['fixed', 'sliding']:
            return jsonify({"error": "window_type必须为'fixed'或'sliding'"}), 400
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 评分
        scores, avg_score, max_score = anomaly_service.score_log_sequence(
            logs, window_type=window_type, stride=stride
        )
        
        # 找出异常窗口
        threshold = data.get('threshold', anomaly_service.threshold)
        anomaly_windows = [i for i, score in enumerate(scores) if score > threshold]
        
        # 返回结果
        return jsonify({
            "scores": [float(s) for s in scores],
            "avg_score": float(avg_score),
            "max_score": float(max_score),
            "threshold": float(threshold),
            "num_windows": len(scores),
            "anomaly_windows": anomaly_windows,
            "num_anomaly_windows": len(anomaly_windows)
        })
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@ai_bp.route('/detect', methods=['POST'])
def detect_anomaly():
    """
    使用异常检测器检测日志
    
    请求体:
    {
        "logs": ["日志1", "日志2", ...],
        "window_type": "sliding",  # 可选，默认为"sliding"
        "stride": 1,               # 可选，默认为1
        "threshold": 0.5           # 可选，默认为0.5
    }
    
    响应:
    {
        "result": {
            "num_windows": 10,
            "avg_score": 0.15,
            "max_score": 0.2,
            "num_anomaly_windows": 1,
            "anomaly_ratio": 0.1,
            "windows": [...]
        }
    }
    """
    try:
        global detector
        
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'logs' not in data:
            return jsonify({"error": "请求体必须包含'logs'字段"}), 400
        
        logs = data['logs']
        window_type = data.get('window_type', 'sliding')
        stride = data.get('stride', 1)
        threshold = data.get('threshold', 0.5)
        
        # 验证窗口类型
        if window_type not in ['fixed', 'sliding']:
            return jsonify({"error": "window_type必须为'fixed'或'sliding'"}), 400
        
        # 确保检测器已初始化
        if detector is None:
            detector = AnomalyDetector()
        
        # 检测日志
        if len(logs) == 1:
            # 单条日志
            result = detector.detect(
                log_text=logs[0],
                threshold=threshold
            )
        else:
            # 日志序列
            result = detector.detect_sequence(
                log_list=logs,
                window_type=window_type,
                stride=stride,
                threshold=threshold
            )
        
        # 返回结果
        return jsonify({"result": result})
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@ai_bp.route('/model/status', methods=['GET'])
def model_status():
    """
    获取模型状态
    
    响应:
    {
        "model_loaded": true,
        "model_path": "/path/to/model.pt"
    }
    """
    try:
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({
                "model_loaded": False,
                "error": "异常检测服务未初始化"
            })
        
        return jsonify({
            "model_loaded": anomaly_service.model is not None,
            "device": str(anomaly_service.device),
            "threshold": float(anomaly_service.threshold),
            "window_size": anomaly_service.window_size,
            "knn_enabled": anomaly_service.knn_model is not None
        })
        
    except Exception as e:
        logger.error(f"获取模型状态时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@ai_bp.route('/model/threshold', methods=['POST'])
def set_threshold():
    """
    设置异常阈值
    
    请求体:
    {
        "threshold": 0.5
    }
    
    响应:
    {
        "success": true,
        "threshold": 0.5
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'threshold' not in data:
            return jsonify({"error": "请求体必须包含'threshold'字段"}), 400
        
        threshold = float(data['threshold'])
        
        # 验证阈值
        if threshold < 0 or threshold > 1:
            return jsonify({"error": "threshold必须在0到1之间"}), 400
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 设置阈值
        anomaly_service.set_threshold(threshold)
        
        # 返回结果
        return jsonify({
            "success": True,
            "threshold": float(threshold)
        })
        
    except Exception as e:
        logger.error(f"设置阈值时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500 