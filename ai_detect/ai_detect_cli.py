#!/usr/bin/env python
import os
import argparse
import logging
import json
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.anomaly_detector import AnomalyDetector


def setup_logging(output_dir=None):
    """设置日志配置"""
    handlers = [logging.StreamHandler()]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, f"detect_{time.strftime('%Y%m%d_%H%M%S')}.log")
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="日志异常检测命令行工具")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--logs", nargs="+", type=str,
                          help="要检测的日志文本列表")
    input_group.add_argument("--log-file", type=str,
                          help="包含日志的文件路径，每行一条日志")
    
    parser.add_argument("--model-path", type=str, required=True,
                      help="预训练模型路径")
    parser.add_argument("--window-type", type=str, choices=['fixed', 'sliding'], default='sliding',
                      help="窗口类型: fixed或sliding")
    parser.add_argument("--window-size", type=int, default=10,
                      help="窗口大小")
    parser.add_argument("--stride", type=int, default=1,
                      help="滑动窗口的步长")
    parser.add_argument("--threshold", type=float, default=0.5,
                      help="异常判定阈值")
    parser.add_argument("--output", type=str, default=None,
                      help="输出结果的JSON文件路径")
    parser.add_argument("--tokenizer-name", type=str, default="prajjwal1/bert-mini",
                      help="使用的tokenizer名称")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    output_dir = os.path.dirname(args.output) if args.output else None
    setup_logging(output_dir)
    
    # 确保模型路径存在
    if not os.path.exists(args.model_path):
        logging.error(f"模型路径不存在: {args.model_path}")
        return 1
    
    # 加载日志
    if args.logs:
        logs = args.logs
    else:
        logs = []
        encodings = ['utf-8', 'latin1', 'cp1252', 'gbk', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(args.log_file, 'r', encoding=encoding) as f:
                    logs = [line.strip() for line in f.readlines() if line.strip()]
                logging.info(f"成功使用 {encoding} 编码读取日志文件")
                break
            except UnicodeDecodeError:
                logging.warning(f"尝试使用 {encoding} 编码读取失败，尝试下一种编码")
            except Exception as e:
                logging.error(f"无法读取日志文件 {args.log_file}: {str(e)}")
                return 1
        
        if not logs:
            error_msg = f"无法使用支持的编码格式读取日志文件 {args.log_file}，尝试了以下编码: {', '.join(encodings)}"
            logging.error(error_msg)
            return 1
    
    if not logs:
        logging.error("没有输入任何日志")
        return 1
    
    logging.info(f"加载了 {len(logs)} 条日志")
    
    # 初始化检测器
    try:
        detector = AnomalyDetector(
            tokenizer_name=args.tokenizer_name,
            window_size=args.window_size
        )
    except Exception as e:
        logging.error(f"初始化检测器失败: {str(e)}")
        return 1
    
    # 开始检测
    try:
        if len(logs) == 1:
            # 单条日志检测
            logging.info("检测单条日志...")
            result = detector.detect(
                log_text=logs[0],
                model_dir=args.model_path,
                threshold=args.threshold
            )
            
            # 输出结果
            print("\n=== 检测结果 ===")
            print(f"日志: {result['log']}")
            print(f"异常分数: {result['score']:.4f}")
            print(f"阈值: {result['threshold']}")
            print(f"结论: {'【异常】' if result['is_anomaly'] else '【正常】'}")
            
        else:
            # 多条日志检测
            logging.info(f"使用 {args.window_type} 窗口检测 {len(logs)} 条日志...")
            result = detector.detect_sequence(
                log_list=logs,
                model_dir=args.model_path,
                window_type=args.window_type,
                stride=args.stride,
                threshold=args.threshold
            )
            
            # 输出结果
            print("\n=== 检测结果摘要 ===")
            print(f"窗口数量: {result['num_windows']}")
            print(f"平均异常分数: {result['avg_score']:.4f}")
            print(f"最大异常分数: {result['max_score']:.4f}")
            print(f"异常窗口数量: {result['num_anomaly_windows']}")
            print(f"异常比例: {result['anomaly_ratio']:.2%}")
            
            # 打印异常窗口
            if result['num_anomaly_windows'] > 0:
                print("\n=== 异常窗口 ===")
                for i, window in enumerate(result['windows']):
                    if window['is_anomaly']:
                        print(f"\n窗口 #{window['window_idx']} (分数: {window['score']:.4f}):")
                        for j, log in enumerate(window['logs']):
                            print(f"  [{window['start_idx'] + j}] {log}")
    
    except Exception as e:
        logging.error(f"检测过程中发生错误: {str(e)}")
        return 1
    
    # 保存结果到文件（如果指定）
    if args.output:
        # 尝试使用相同的编码格式保存结果
        successful_encoding = 'utf-8'  # 默认编码
        for encoding in ['utf-8', 'latin1', 'cp1252', 'gbk', 'iso-8859-1']:
            try:
                # 尝试读取一行，检查是否可以用这种编码打开文件
                if os.path.exists(args.log_file):
                    with open(args.log_file, 'r', encoding=encoding) as f:
                        next(f, None)
                    successful_encoding = encoding
                    break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        try:
            with open(args.output, 'w', encoding=successful_encoding) as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logging.info(f"结果已保存到: {args.output} (使用 {successful_encoding} 编码)")
        except Exception as e:
            logging.error(f"保存结果失败: {str(e)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 