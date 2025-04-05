import os
import argparse
import logging
import json
import time
from anomaly_detector import AnomalyDetector


def setup_logging(output_dir):
    """设置日志配置"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"evaluate_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="评估日志异常检测模型")
    
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试数据文件路径（包含标签）")
    parser.add_argument("--model_path", type=str, required=True,
                        help="预训练模型路径")
    parser.add_argument("--output_dir", type=str, default="./ai_detect/output",
                        help="输出结果目录")
    parser.add_argument("--tokenizer_name", type=str, default="prajjwal1/bert-mini",
                        help="使用的tokenizer名称")
    parser.add_argument("--window_size", type=int, default=10,
                        help="日志窗口大小")
    parser.add_argument("--threshold", type=float, default=None,
                        help="异常判定阈值（可选，不提供则自动寻找最佳阈值）")
    parser.add_argument("--detection_method", type=str, default="ensemble",
                        choices=["knn", "cluster", "lof", "iforest", "reconstruction", "ensemble"],
                        help="无监督异常检测方法")
    parser.add_argument("--eval_all_methods", action="store_true",
                        help="是否评估所有检测方法")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(args.output_dir)
    
    # 记录参数
    logging.info("评估参数:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    
    # 确保模型路径存在
    if not os.path.exists(args.model_path):
        logging.error(f"模型路径不存在: {args.model_path}")
        return
    
    # 初始化检测器
    detector = AnomalyDetector(
        model_dir=os.path.dirname(args.model_path),
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        window_size=args.window_size,
        detection_method=args.detection_method
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
                
        results = detector.evaluate(
            test_file=args.test_file,
            model_path=args.model_path,
            threshold=args.threshold,
            eval_methods=eval_methods
        )
        
        # 保存评估结果
        results_file = os.path.join(args.output_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
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
        logging.info(f"评估结果已保存到: {results_file}")
        
    except Exception as e:
        logging.error(f"评估过程中发生错误: {str(e)}")
        raise
    
    logging.info("评估完成!")


if __name__ == "__main__":
    main() 