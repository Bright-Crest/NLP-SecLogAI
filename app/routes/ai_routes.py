from flask import Blueprint, request, jsonify, current_app, render_template
import os
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.anomaly_score import get_anomaly_service, init_anomaly_service

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT_DIR, 'ai_detect', 'checkpoint')

# 创建蓝图
ai_bp = Blueprint('ai', __name__, url_prefix='/ai')

# 设置跨域资源共享(CORS)
@ai_bp.after_request
def add_cors_headers(response):
    """为所有响应添加CORS头信息"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

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


@ai_bp.route('/model/threshold', methods=['POST', 'OPTIONS'])
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
    # 处理OPTIONS请求（用于CORS预检）
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
        
    try:
        # 记录原始请求数据，用于调试
        logger.info(f"收到阈值设置请求 - Headers: {dict(request.headers)}")
        raw_data = request.get_data(as_text=True)
        logger.info(f"阈值设置请求内容: {raw_data}")
        
        # 从不同可能的源获取数据
        if request.is_json:
            data = request.get_json()
            logger.info(f"解析的JSON数据: {data}")
        elif request.form:
            data = request.form.to_dict()
            logger.info(f"表单数据: {data}")
        else:
            # 尝试手动解析JSON
            try:
                import json
                data = json.loads(raw_data)
                logger.info(f"手动解析的JSON: {data}")
            except Exception as parse_err:
                logger.error(f"解析请求数据失败: {str(parse_err)}")
                data = {}
        
        # 验证数据
        if not data:
            return jsonify({"error": "请求体为空"}), 400
            
        if 'threshold' not in data:
            return jsonify({"error": "请求体必须包含'threshold'字段"}), 400
        
        # 处理不同类型的阈值输入
        try:
            threshold = float(data['threshold'])
        except (ValueError, TypeError) as ve:
            logger.error(f"阈值转换错误: {str(ve)}, 原始值: {data['threshold']}")
            return jsonify({"error": f"阈值必须是有效的数字，而不是 {data['threshold']}"}), 400
        
        # 验证阈值范围
        if threshold < 0 or threshold > 1:
            return jsonify({"error": f"threshold必须在0到1之间，当前值: {threshold}"}), 400
        
        # 获取异常服务
        anomaly_service = get_anomaly_service()
        
        if anomaly_service is None:
            return jsonify({"error": "异常检测服务未初始化"}), 500
        
        # 记录原始阈值，用于验证更改是否成功
        original_threshold = anomaly_service.threshold
        logger.info(f"当前阈值: {original_threshold}, 将设置为: {threshold}")
        
        # 设置阈值
        anomaly_service.set_threshold(threshold)
        
        # 验证阈值是否真正更改
        new_threshold = anomaly_service.threshold
        logger.info(f"阈值设置后的值: {new_threshold}")
        
        if abs(new_threshold - threshold) > 1e-6:  # 浮点数比较
            logger.warning(f"阈值设置可能未生效，期望值: {threshold}, 实际值: {new_threshold}")
        
        # 构建响应
        response = jsonify({
            "success": True,
            "threshold": float(new_threshold),
            "previous_threshold": float(original_threshold)
        })
        
        # 添加CORS头
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    except Exception as e:
        logger.error(f"设置阈值时发生错误: {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "details": str(e.__class__.__name__)}), 500


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


# 添加Web界面路由
@ai_bp.route('/ui', methods=['GET'])
def ai_detector_ui():
    """
    AI异常检测Web界面
    
    提供基于Bootstrap的前端界面，用于直观地进行日志异常检测
    """
    return render_template('ai_detector.html')

@ai_bp.route('/debug/threshold', methods=['GET', 'POST'])
def debug_threshold():
    """
    调试专用接口：直接设置/获取阈值，方便前端调试
    
    GET: 返回当前阈值
    POST: 接收表单或JSON格式设置阈值
    
    注意：仅用于测试环境
    """
    anomaly_service = get_anomaly_service()
    
    if anomaly_service is None:
        return jsonify({"error": "异常检测服务未初始化"}), 500
        
    # GET请求：返回当前阈值
    if request.method == 'GET':
        return jsonify({
            "current_threshold": float(anomaly_service.threshold),
            "service_initialized": True
        })
    
    # POST请求：设置新阈值
    try:
        # 尝试从不同来源获取阈值
        threshold = None
        
        # 从JSON获取
        if request.is_json:
            data = request.get_json()
            threshold = data.get('threshold')
            
        # 从Form表单获取    
        elif request.form:
            threshold = request.form.get('threshold')
            
        # 从URL参数获取
        else:
            threshold = request.args.get('threshold')
            
        if threshold is None:
            return jsonify({"error": "无法获取阈值参数，请通过JSON、表单或URL参数提供"}), 400
            
        # 转换为浮点数
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            return jsonify({"error": f"无效的阈值格式: {threshold}，必须是有效数字"}), 400
            
        # 记录设置前的阈值
        old_threshold = anomaly_service.threshold
        
        # 设置新阈值
        success = anomaly_service.set_threshold(threshold)
        
        if success:
            return jsonify({
                "success": True,
                "old_threshold": float(old_threshold),
                "new_threshold": float(anomaly_service.threshold),
                "message": f"阈值已从 {old_threshold} 更新为 {anomaly_service.threshold}"
            })
        else:
            return jsonify({
                "success": False,
                "threshold": float(old_threshold),
                "message": "阈值设置失败，可能是因为值无效"
            }), 400
            
    except Exception as e:
        logger.error(f"调试阈值接口错误: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "type": str(e.__class__.__name__)
        }), 500 