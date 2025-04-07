import os
import sys
import argparse
import logging
import json
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "app", "checkpoint")
OUTPUT_DIR = os.path.join(ROOT_DIR, "ai_detect", "output")

sys.path.append(ROOT_DIR)
from app.ai_models.anomaly_detector import AnomalyDetector


def setup_logging(output_dir):
    """设置日志配置"""
    try:
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置清晰的日志文件名
        log_file = os.path.join(output_dir, f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}.log")
        
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估日志异常检测模型")
    
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件（包含标签或原始日志文件）")
    parser.add_argument("--model_dir", type=str, default=MODEL_DIR,
                        help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="输出结果目录")
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-mini",
                        help="使用的tokenizer名称")
    parser.add_argument("--window_size", type=int, default=10,
                        help="日志窗口大小")
    parser.add_argument("--threshold", type=float, default=None,
                        help="异常判定阈值（可选，不提供则自动寻找最佳阈值）")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批量处理窗口大小")
    parser.add_argument("--detection_method", type=str, default="ensemble",
                        choices=["knn", "cluster", "lof", "iforest", "reconstruction", "ensemble"],
                        help="无监督异常检测方法")
    parser.add_argument("--eval_all_methods", action="store_true",
                        help="是否评估所有检测方法")
    parser.add_argument("--with_lables", action="store_true",
                        help="指定测试数据没有标签，使用无监督评估")
    parser.add_argument("--top_k", type=int, default=20,
                        help="无监督评估时显示的Top-K异常样本数量")
    
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
    logging.info("评估参数:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    
    # 确保模型路径存在
    if not os.path.exists(args.model_dir):
        logging.error(f"模型路径不存在: {args.model_dir}")
        return
    
    # 初始化检测器
    detector = AnomalyDetector(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        window_size=args.window_size,
        detection_method=args.detection_method,
        load_from_model_dir=True
    )
    
    # 开始评估
    logging.info("开始评估模型...")
    start_time = time.time()
    
    try:
        # 确定评估方法
        eval_methods = None
        if args.eval_all_methods:
            # 评估所有方法，包括融合方法
            eval_methods = detector.DETECTION_METHODS
            logging.info(f"将评估所有方法: {', '.join(eval_methods)}")
        else:
            # 只评估指定方法，如果是ensemble则加上单一方法进行对比
            if args.detection_method == 'ensemble':
                # 评估所有单一方法
                eval_methods = detector.DETECTION_METHODS
                logging.info(f"将评估融合方法及所有单一方法")
            else:
                # 只评估指定方法和融合方法
                eval_methods = [args.detection_method, 'ensemble']
                logging.info(f"将评估指定方法({args.detection_method})和融合方法")
        
        # 设置是否使用无监督评估
        with_lables = args.with_lables
        if not with_lables:
            logging.info("使用无监督评估模式（无标签）")
        else:
            logging.info("使用有监督评估模式（带标签）")
        
        # 执行评估
        results = detector.evaluate(
            test_file=args.test_file,
            model_dir=args.model_dir,
            threshold=args.threshold,
            eval_methods=eval_methods,
            no_labels=not with_lables,
            batch_size=args.batch_size
        )
        
        # 打印结果概要
        if not with_lables:
            # 无监督评估结果
            logging.info(f"无监督评估结果:")
            logging.info(f"  主方法: {results.get('detection_method', args.detection_method)}")
            logging.info(f"  推荐阈值: {results.get('threshold', 0.0):.4f}")
            logging.info(f"  样本数量: {results.get('num_samples', 0)}")
            logging.info(f"  可能异常样本数量: {results.get('num_anomalies', 0)} ({results.get('anomaly_ratio', 0.0)*100:.2f}%)")
            
            # 打印模型评估等级
            if 'summary' in results:
                summary = results['summary']
                logging.info(f"  模型评估: {summary.get('grade', '未知')} (分数: {summary.get('score', 0)})")
                logging.info(f"  评估摘要: {summary.get('summary', '无')}")
                
                # 打印评估因素
                if 'factors' in summary and summary['factors']:
                    logging.info("  评估因素:")
                    for factor in summary['factors']:
                        logging.info(f"    - {factor}")
            
            # 打印建议
            if 'suggestions' in results and results['suggestions']:
                logging.info("  建议:")
                for suggestion in results['suggestions']:
                    logging.info(f"    - {suggestion}")
            
            # 如果评估了多个方法，输出所有方法的质量
            if 'all_methods' in results:
                logging.info("  各方法评估:")
                for method, quality in results['all_methods'].items():
                    logging.info(f"    {method}: {quality}")
            
            # 如果使用了融合方法，输出权重
            if 'ensemble_weights' in results:
                logging.info("  融合器权重:")
                for method, weight in results['ensemble_weights'].items():
                    logging.info(f"    {method}: {weight:.4f}")
        else:
            # 有监督评估结果
            logging.info(f"评估结果:")
            logging.info(f"  主方法({results.get('detection_method', args.detection_method)}) AUC: {results['auc']:.4f}")
            logging.info(f"  最佳阈值: {results['threshold']:.4f}")
            logging.info(f"  样本数量: {results['num_samples']}")
            logging.info(f"  异常样本数量: {results['num_anomalies']}")
            logging.info(f"  准确率: {results['accuracy']:.4f}")
            
            # 如果评估了多个方法，输出所有方法的AUC
            if 'all_methods' in results:
                logging.info("  各方法AUC值:")
                # 按AUC值排序
                sorted_methods = sorted(results['all_methods'].items(), key=lambda x: x[1], reverse=True)
                for method, auc in sorted_methods:
                    logging.info(f"    {method}: {auc:.4f}")
            
            # 如果使用了融合方法，输出融合器的性能摘要
            if 'ensemble_summary' in results:
                summary = results['ensemble_summary']
                if 'detector_weights' in summary:
                    logging.info("  融合器权重:")
                    for method, weight in summary['detector_weights'].items():
                        logging.info(f"    {method}: {weight:.4f}")
        
        logging.info(f"评估耗时: {(time.time() - start_time):.2f} 秒")
        
    except Exception as e:
        logging.error(f"评估过程中发生错误: {str(e)}")
        raise
    
    logging.info("评估完成!")


if __name__ == "__main__":
    main() 