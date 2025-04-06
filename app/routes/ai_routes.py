from flask import Blueprint, request, jsonify, current_app
import os
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.anomaly_score import get_anomaly_service, init_anomaly_service

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT_DIR, 'ai_detect', 'checkpoint')

# 创建蓝图
ai_bp = Blueprint('ai', __name__, url_prefix='/ai')

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化AI服务的函数
def setup_ai_services():
    """初始化AI检测服务"""
    # 默认模型路径
    model_dir = os.environ.get('AI_MODEL_PATH', MODEL_DIR)
    
    # 检查模型是否存在
    if os.path.exists(model_dir):
        # 初始化异常评分服务
        init_anomaly_service(model_dir=model_dir)
        logger.info(f"异常评分服务已初始化，使用模型: {model_dir}")
    else:
        logger.warning(f"模型文件不存在: {model_dir}")

# 定义初始化函数，根据Flask版本提供兼容性
def init_ai_bp(app=None):
    """
    初始化AI蓝图
    在注册蓝图时调用该函数 
    例如: init_ai_bp(app)
    """
    if app is not None:
        # 检查Flask版本
        try:
            # 尝试使用before_first_request (Flask 2.0之前的版本)
            app.before_first_request(setup_ai_services)
            logger.info("使用app.before_first_request注册AI服务初始化")
        except AttributeError:
            # 对于Flask 2.0+版本，手动设置一个应用级别的before_request
            @app.before_request
            def init_before_first_request():
                if not hasattr(app, '_ai_services_initialized'):
                    setup_ai_services()
                    setattr(app, '_ai_services_initialized', True)
            logger.info("使用app.before_request钩子注册AI服务初始化")
    else:
        # 如果没有传递app参数，那么直接初始化服务
        setup_ai_services()
        logger.info("直接初始化AI服务")

# 注册路由函数，供测试和应用使用
def register_routes(app):
    """
    在Flask应用中注册AI蓝图并初始化服务
    
    参数:
        app: Flask应用实例
    """
    # 先初始化AI服务（包括设置before_request钩子）
    init_ai_bp(app)
    
    # 然后注册蓝图
    app.register_blueprint(ai_bp)
    
    return app

@ai_bp.route('/score_log', methods=['POST'])
def score_log():
    """
    对单条日志进行异常评分
    
    请求体:
    {
        "log": "日志文本",
        "use_knn": true  # 可选，是否使用KNN增强
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
        use_knn = data.get('use_knn', None)  # 使用None表示使用服务默认设置
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 计算异常分数
        score = anomaly_service.score_single_log(log_text, use_knn=use_knn)
        
        # 判断是否异常
        threshold = data.get('threshold', anomaly_service.threshold)
        is_anomaly = anomaly_service.is_anomaly(score)
        
        # 返回结果
        return jsonify({
            "log": log_text,
            "score": float(score),
            "threshold": float(threshold),
            "is_anomaly": bool(is_anomaly),
            "knn_used": use_knn if use_knn is not None else (anomaly_service.use_knn and anomaly_service.knn_model is not None)
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
        "stride": 1,             # 可选，默认为1
        "use_knn": true          # 可选，是否使用KNN增强
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
        use_knn = data.get('use_knn', None)  # 使用None表示使用服务默认设置
        
        # 验证窗口类型
        if window_type not in ['fixed', 'sliding']:
            return jsonify({"error": "window_type必须为'fixed'或'sliding'"}), 400
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 评分
        scores, avg_score, max_score = anomaly_service.score_log_sequence(
            logs, window_type=window_type, stride=stride, use_knn=use_knn
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
            "num_anomaly_windows": len(anomaly_windows),
            "knn_used": anomaly_service.use_knn and anomaly_service.knn_model is not None
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
        "threshold": 0.5,          # 可选，默认为0.5
        "use_knn": true            # 可选，是否使用KNN增强
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
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'logs' not in data:
            return jsonify({"error": "请求体必须包含'logs'字段"}), 400
        
        logs = data['logs']
        window_type = data.get('window_type', 'sliding')
        stride = data.get('stride', 1)
        threshold = data.get('threshold', 0.5)
        use_knn = data.get('use_knn', None)  # 使用None表示使用服务默认设置
        
        # 验证窗口类型
        if window_type not in ['fixed', 'sliding']:
            return jsonify({"error": "window_type必须为'fixed'或'sliding'"}), 400
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 检测日志
        if len(logs) == 1:
            # 单条日志
            score = anomaly_service.score_single_log(logs[0], use_knn=use_knn)
            is_anomaly = anomaly_service.is_anomaly(score)
            
            result = {
                "log": logs[0],
                "score": float(score),
                "threshold": float(threshold),
                "is_anomaly": bool(is_anomaly),
                "knn_used": use_knn if use_knn is not None else (anomaly_service.use_knn and anomaly_service.knn_model is not None)
            }
        else:
            # 日志序列
            try:
                # 尝试调用score_log_sequence
                scores_result = anomaly_service.score_log_sequence(
                    logs, window_type=window_type, stride=stride, use_knn=use_knn
                )
                
                # 确保返回了三个值
                if not scores_result or len(scores_result) != 3:
                    scores, avg_score, max_score = [], 0.0, 0.0
                else:
                    scores, avg_score, max_score = scores_result
                
                # 根据阈值识别异常窗口
                windows = []
                for i, score in enumerate(scores):
                    # 对于滑动窗口，记录窗口开始位置
                    start_idx = i * stride if window_type == 'sliding' else i * anomaly_service.window_size
                    end_idx = min(start_idx + anomaly_service.window_size, len(logs))
                    window_logs = logs[start_idx:end_idx]
                    
                    windows.append({
                        "window_idx": i,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "logs": window_logs,
                        "score": float(score),
                        "is_anomaly": score > threshold
                    })
                
                # 找出异常窗口
                anomaly_windows = [w for w in windows if w["is_anomaly"]]
                
                result = {
                    "num_windows": len(windows),
                    "avg_score": float(avg_score),
                    "max_score": float(max_score),
                    "num_anomaly_windows": len(anomaly_windows),
                    "anomaly_ratio": float(len(anomaly_windows) / len(windows)) if windows else 0.0,
                    "windows": windows,
                    "knn_used": use_knn if use_knn is not None else (anomaly_service.use_knn and anomaly_service.knn_model is not None)
                }
            except Exception as inner_e:
                logger.error(f"评分日志序列时发生错误: {str(inner_e)}")
                # 返回一个空的结果结构
                result = {
                    "num_windows": 0,
                    "avg_score": 0.0,
                    "max_score": 0.0,
                    "num_anomaly_windows": 0,
                    "anomaly_ratio": 0.0,
                    "windows": [],
                    "error": str(inner_e),
                    "knn_used": False
                }
        
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
            "batch_size": anomaly_service.batch_size,
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


@ai_bp.route('/knn/build', methods=['POST'])
def build_knn():
    """
    构建KNN嵌入库
    
    请求体:
    {
        "normal_logs": ["正常日志1", "正常日志2", ...],
    }
    
    响应:
    {
        "success": true,
        "num_embeddings": 100,
        "message": "成功构建KNN嵌入库"
    }
    """
    try:
        # 获取请求数据
        data = request.get_json()
        
        if not data or 'normal_logs' not in data:
            return jsonify({"error": "请求体必须包含'normal_logs'字段"}), 400
        
        normal_logs = data['normal_logs']
        
        if len(normal_logs) < 10:
            return jsonify({"error": "正常日志数量不足，至少需要10条"}), 400
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 构建嵌入库
        success = anomaly_service.build_embedding_bank(normal_logs)
        
        if success:
            return jsonify({
                "success": True,
                "num_embeddings": len(anomaly_service.embeddings_bank) if anomaly_service.embeddings_bank is not None else 0,
                "message": "成功构建KNN嵌入库"
            })
        else:
            return jsonify({
                "success": False,
                "message": "KNN嵌入库构建失败"
            })
            
    except Exception as e:
        logger.error(f"构建KNN嵌入库时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500


@ai_bp.route('/knn/status', methods=['GET', 'POST'])
def knn_status():
    """
    获取或设置KNN增强状态
    
    GET请求:
    获取当前KNN状态
    
    POST请求体:
    {
        "enabled": true  # 开启或关闭KNN增强
    }
    
    响应:
    {
        "knn_enabled": true,
        "num_embeddings": 100
    }
    """
    try:
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # POST请求用于设置状态
        if request.method == 'POST':
            data = request.get_json()
            
            if not data or 'enabled' not in data:
                return jsonify({"error": "请求体必须包含'enabled'字段"}), 400
            
            enabled = data['enabled']
            
            # 如果要启用KNN但嵌入库未构建，则返回错误
            if enabled and anomaly_service.knn_model is None:
                return jsonify({
                    "error": "无法启用KNN增强，嵌入库未构建，请先调用/ai/knn/build接口"
                }), 400
            
            # 设置KNN增强状态
            anomaly_service.set_use_knn(enabled)
            
            # 返回成功响应，包含success字段
            return jsonify({
                "success": True,
                "knn_enabled": anomaly_service.use_knn,
                "knn_available": anomaly_service.knn_model is not None,
                "num_embeddings": len(anomaly_service.embeddings_bank) if anomaly_service.embeddings_bank is not None else 0
            })
        
        # 返回当前状态
        return jsonify({
            "knn_enabled": anomaly_service.use_knn,
            "knn_available": anomaly_service.knn_model is not None,
            "num_embeddings": len(anomaly_service.embeddings_bank) if anomaly_service.embeddings_bank is not None else 0
        })
            
    except Exception as e:
        logger.error(f"处理KNN状态请求时发生错误: {str(e)}")
        return jsonify({"error": str(e)}), 500 