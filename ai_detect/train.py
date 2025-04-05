import os
import sys
import argparse
import logging
import time
import random
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.anomaly_detector import AnomalyDetector

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "ai_detect", "checkpoint")
OUTPUT_DIR = os.path.join(ROOT_DIR, "ai_detect", "output")


def setup_logging(output_dir):
    """设置日志配置"""
    try:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置清晰的日志文件名
        log_file = os.path.join(output_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
        
        # 尝试创建一个空的日志文件，检查写入权限
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"# 日志开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 检查文件是否被创建
        if not os.path.exists(log_file):
            print(f"警告: 无法创建日志文件 {log_file}")
            return False
            
        # 配置根日志记录器
        root_logger = logging.getLogger()
        # 重置处理程序，避免重复
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                
        # 创建并配置文件处理程序
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        
        # 创建并配置控制台处理程序
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # 添加处理程序到根日志记录器
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.INFO)
        
        # 添加初始日志条目，确认日志系统已启动
        logging.info(f"日志系统已初始化，日志文件: {log_file}")
        
        return True
    except Exception as e:
        print(f"设置日志系统时发生错误: {str(e)}")
        # 设置基本的控制台日志以便继续运行
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.warning(f"无法设置文件日志，将只使用控制台日志: {str(e)}")
        return False


def split_logs_for_eval(log_file, eval_ratio=0.2, min_eval_lines=100):
    """
    从日志文件中划分评估数据，保持时序性
    
    参数:
        log_file: 日志文件路径
        eval_ratio: 评估数据比例
        min_eval_lines: 最小评估数据行数
        
    返回:
        train_file: 训练数据文件路径
        eval_file: 评估数据文件路径
    """
    # 读取日志文件
    logs = []
    encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                logs = [line.strip() for line in f.readlines() if line.strip()]
            logging.info(f"成功使用 {encoding} 编码读取日志文件")
            break
        except UnicodeDecodeError:
            logging.warning(f"尝试使用 {encoding} 编码读取失败，尝试下一种编码")
        except Exception as e:
            logging.error(f"读取日志文件 {log_file} 失败: {str(e)}")
            raise
    
    if not logs:
        error_msg = f"无法使用支持的编码格式读取日志文件 {log_file}，尝试了以下编码: {', '.join(encodings)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    total_lines = len(logs)
    logging.info(f"从 {log_file} 读取了 {total_lines} 行日志")
    
    if total_lines == 0:
        raise ValueError(f"日志文件 {log_file} 为空或格式不正确")
    
    # 计算评估数据行数，确保有足够的评估数据
    eval_lines = max(min_eval_lines, int(total_lines * eval_ratio))
    
    # 确保评估数据不超过总数的一半
    eval_lines = min(eval_lines, total_lines // 2)
    
    # 确保训练数据足够
    if total_lines - eval_lines < min_eval_lines:
        logging.warning(f"总数据行数不足，减少评估数据比例")
        eval_lines = total_lines // 4  # 最多使用25%的数据作为评估
    
    # 随机决定从开头还是结尾选择评估数据
    use_beginning = random.random() < 0.5
    if use_beginning:
        # 从开头选择
        logging.info(f"从日志开头选择 {eval_lines}/{total_lines} 行作为评估数据 ({eval_lines/total_lines:.1%})")
        eval_logs = logs[:eval_lines]
        train_logs = logs[eval_lines:]
    else:
        # 从结尾选择
        logging.info(f"从日志结尾选择 {eval_lines}/{total_lines} 行作为评估数据 ({eval_lines/total_lines:.1%})")
        eval_logs = logs[-eval_lines:]
        train_logs = logs[:-eval_lines]
    
    # 将评估数据保存到临时文件
    base_dir = os.path.dirname(log_file)
    base_name = os.path.basename(log_file)
    name_without_ext, ext = os.path.splitext(base_name)
    
    eval_file = os.path.join(base_dir, f"{name_without_ext}_eval{ext}")
    train_file = os.path.join(base_dir, f"{name_without_ext}_train{ext}")
    
    # 获取成功读取文件的编码
    successful_encoding = encodings[0]  # 默认第一个编码
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                # 尝试读取一行，验证编码是否工作
                next(f, None)
                successful_encoding = encoding
                break
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    
    try:
        with open(eval_file, 'w', encoding=successful_encoding) as f:
            for line in eval_logs:
                f.write(f"{line}\n")
        
        with open(train_file, 'w', encoding=successful_encoding) as f:
            for line in train_logs:
                f.write(f"{line}\n")
    except Exception as e:
        logging.error(f"保存划分后的数据文件失败: {str(e)}")
        raise
    
    logging.info(f"训练数据: {len(train_logs)} 行 ({len(train_logs)/total_lines:.1%}), 保存到: {train_file}")
    logging.info(f"评估数据: {len(eval_logs)} 行 ({len(eval_logs)/total_lines:.1%}), 保存到: {eval_file}")
    
    return train_file, eval_file


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练日志异常检测模型")
    
    # 基本参数
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件路径")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="评估数据文件路径（可选）")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR,
                        help="模型保存目录")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="输出结果目录")
    
    # 模型参数
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-mini",
                        help="使用的tokenizer名称")
    parser.add_argument("--window_size", type=int, default=10,
                        help="日志窗口大小")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="保存模型的步数间隔")
    parser.add_argument("--eval_ratio", type=float, default=0.2,
                        help="自动划分时评估数据比例（当未提供eval_file时使用）")
    
    # 新增参数：早停和TensorBoard
    parser.add_argument("--not_early_stopping", action="store_true", 
                        help="是否不启用早停机制")
    parser.add_argument("--patience", type=int, default=3,
                        help="早停的耐心值（连续多少次评估没有改进后停止）")
    parser.add_argument("--not_use_tensorboard", action="store_true",
                        help="是否不使用TensorBoard记录训练过程")
    
    # 无监督异常检测方法选择
    parser.add_argument("--detection_method", type=str, default="ensemble",
                        choices=["knn", "cluster", "lof", "iforest", "reconstruction", "ensemble"],
                        help="无监督异常检测方法")
    parser.add_argument("--fusion_method", type=str, default="dynamic_weight",
                        choices=["dynamic_weight", "static_weight", "max", "mean"],
                        help="异常分数融合方法，用于ensemble检测方法")
    parser.add_argument("--eval_all_methods", action="store_true",
                        help="评估时是否测试所有检测方法")
    parser.add_argument("--not_eval_ensemble", action="store_true",
                        help="评估时是否不评估融合模式（如果使用单一检测方法）")
    parser.add_argument("--not_use_contrastive", action="store_true",
                        help="不使用对比学习")
    parser.add_argument("--not_use_augmentation", action="store_true",
                        help="不使用数据增强")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    logs_dir = os.path.join(args.output_dir, "logs")
    logging_success = setup_logging(logs_dir)
    
    if not logging_success:
        logging.warning(f"警告: 日志系统设置失败，将继续执行但日志可能不会被正确保存")
    
    # 确保输出目录存在
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"创建输出目录失败: {str(e)}")
        return
    
    # 记录参数
    logging.info("训练参数:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    
    train_file = args.train_file
    eval_file = args.eval_file
    
    # 检查训练文件是否存在
    if not os.path.exists(train_file):
        logging.error(f"训练文件 {train_file} 不存在")
        return
    
    # 如果未提供评估文件，则自动划分
    if eval_file is None or not os.path.exists(eval_file):
        logging.info("未提供评估数据文件或指定文件不存在，将从训练数据自动划分评估数据")
        try:
            train_file, eval_file = split_logs_for_eval(
                train_file, 
                eval_ratio=args.eval_ratio
            )
        except Exception as e:
            logging.error(f"自动划分数据失败: {str(e)}")
            logging.warning("无法进行有效的评估，将仅使用训练数据进行训练")
            eval_file = None
            
            # 如果需要早停，提示用户
            if not args.not_early_stopping:
                logging.warning("早停功能需要评估数据，这些功能将被禁用")
    
    # 确定是否使用早停
    use_early_stopping = not args.not_early_stopping and eval_file is not None
    if use_early_stopping:
        logging.info(f"启用早停机制，耐心值: {args.patience}")
    else:
        logging.info("未启用早停机制")
    
    # 确定是否使用TensorBoard
    if not args.not_use_tensorboard:
        logging.info("启用TensorBoard记录")
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logging.info(f"TensorBoard日志将保存在: {tensorboard_dir}")
        logging.info(f"启动TensorBoard: tensorboard --logdir={tensorboard_dir}")
    
    # 初始化检测器
    detector = AnomalyDetector(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        window_size=args.window_size,
        detection_method=args.detection_method
    )
    
    # 确定对比学习和数据增强设置
    enable_contrastive = not args.not_use_contrastive
    enable_augmentation = not args.not_use_augmentation
    
    # 开始训练
    logging.info("开始训练模型...")
    logging.info(f"训练数据: {train_file}")
    logging.info(f"评估数据: {eval_file if eval_file else '无'}")
    logging.info(f"异常检测方法: {args.detection_method}")
    if args.detection_method == 'ensemble':
        logging.info(f"融合方法: {args.fusion_method}")
    logging.info(f"对比学习: {'启用' if enable_contrastive else '禁用'}")
    logging.info(f"数据增强: {'启用' if enable_augmentation else '禁用'}")
    
    start_time = time.time()
    
    try:
        # 训练模型
        detector.train(
            train_file=train_file,
            eval_file=eval_file,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            save_steps=args.save_steps,
            early_stopping_patience=args.patience if use_early_stopping else 0,
            use_tensorboard=not args.not_use_tensorboard,
            enable_contrastive=enable_contrastive,
            enable_augmentation=enable_augmentation,
            fusion_method=args.fusion_method
        )
        
        # 进行评估（如果有评估数据）
        if eval_file is not None:
            logging.info("对训练好的模型进行评估...")
            
            # 决定评估哪些方法
            eval_methods = None  # 默认当前方法
            if args.eval_all_methods:
                # 使用所有单一方法，不包括ensemble
                eval_methods = [m for m in detector.DETECTION_METHODS if m != 'ensemble']
                logging.info(f"将评估所有检测方法: {', '.join(eval_methods)}")
            
            # 进行评估
            eval_results = detector.evaluate(
                test_file=eval_file, 
                model_dir=args.model_dir,
                eval_methods=eval_methods,
                eval_ensemble=not args.not_eval_ensemble,
                no_labels=True  # 使用无监督评估，即使可能有标签也以无标签方式评估
            )
            
            # 输出评估结果
            logging.info(f"无监督评估结果:")
            logging.info(f"  主方法: {eval_results.get('detection_method', args.detection_method)}")
            logging.info(f"  推荐阈值: {eval_results.get('threshold', 0.0):.4f}")
            logging.info(f"  样本数量: {eval_results.get('num_samples', 0)}")
            logging.info(f"  可能异常样本数量: {eval_results.get('num_anomalies', 0)} ({eval_results.get('anomaly_ratio', 0.0)*100:.2f}%)")
            
            # 打印模型评估等级
            if 'summary' in eval_results:
                summary = eval_results['summary']
                logging.info(f"  模型评估: {summary.get('grade', '未知')} (分数: {summary.get('score', 0)})")
                logging.info(f"  评估摘要: {summary.get('summary', '无')}")
                
                # 打印评估因素
                if 'factors' in summary and summary['factors']:
                    logging.info("  评估因素:")
                    for factor in summary['factors']:
                        logging.info(f"    - {factor}")
            
            # 打印建议
            if 'suggestions' in eval_results and eval_results['suggestions']:
                logging.info("  建议:")
                for suggestion in eval_results['suggestions']:
                    logging.info(f"    - {suggestion}")
            
            # 如果评估了多个方法，输出所有方法的质量
            if 'all_methods' in eval_results:
                logging.info("  各方法评估:")
                for method, quality in eval_results['all_methods'].items():
                    logging.info(f"    {method}: {quality}")
            
            # 如果使用了融合方法，输出权重
            if 'ensemble_weights' in eval_results:
                logging.info("  融合器权重:")
                for method, weight in eval_results['ensemble_weights'].items():
                    logging.info(f"    {method}: {weight:.4f}")
            
        logging.info(f"训练耗时: {(time.time() - start_time) / 60:.2f} 分钟")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {str(e)}")
        raise
    
    logging.info("训练完成!")


if __name__ == "__main__":
    main() 