#!/usr/bin/env python3
import argparse
import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.anomaly_score import get_anomaly_score, is_log_anomaly
from app.services.log_tokenizer import LogTokenizer
from utils.log_converter import LogConverter # 导入日志转换器


def is_json_file(file_path):
    """检查文件是否为JSON格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            json.loads(line)
            return True
    except (json.JSONDecodeError, UnicodeDecodeError, IOError):
        return False

def main():
    """TinyLogBERT CLI工具 - 用于命令行日志异常检测"""
    parser = argparse.ArgumentParser(description="TinyLogBERT - 无盒标日志异常检测工具")
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 单日志检测子命令
    detect_parser = subparsers.add_parser("detect", help="检测单条日志是否异常")
    detect_parser.add_argument("log", help="日志文本")
    detect_parser.add_argument("-t", "--threshold", type=float, help="异常检测阈值(0-1)")
    
    # 批量检测子命令
    batch_parser = subparsers.add_parser("batch", help="批量检测日志文件中的日志")
    batch_parser.add_argument("file", help="日志文件路径")
    batch_parser.add_argument("-t", "--threshold", type=float, help="异常检测阈值(0-1)")
    batch_parser.add_argument("-o", "--output", help="输出结果文件路径")
    batch_parser.add_argument("--json-format", action="store_true", help="指定输入文件为JSON格式（每行一个JSON）")
    
    # 分词子命令
    tokenize_parser = subparsers.add_parser("tokenize", help="将日志转换为token列表")
    tokenize_parser.add_argument("log", help="日志文本")
    
    # 日志转换子命令
    if LogConverter:
        convert_parser = subparsers.add_parser("convert", help="将普通日志文件转换为JSON格式")
        convert_parser.add_argument("file", help="日志文件路径")
        convert_parser.add_argument("-o", "--output", help="输出JSON文件路径")
        convert_parser.add_argument("-r", "--anomaly-ratio", type=float, default=0.05, help="模拟的异常比例(0-1)，默认0.05")
        convert_parser.add_argument("-f", "--force", action="store_true", help="强制转换，即使输出文件已存在")
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定子命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 处理各个子命令
    if args.command == "detect":
        detect_single_log(args.log, args.threshold)
    elif args.command == "batch":
        batch_detect_logs(args.file, args.threshold, args.output, args.json_format if hasattr(args, "json_format") else False)
    elif args.command == "tokenize":
        tokenize_log(args.log)
    elif args.command == "convert" and LogConverter:
        convert_log_file(args.file, args.output, args.anomaly_ratio, args.force)

def detect_single_log(log_text, threshold=None):
    """检测单条日志"""
    try:
        # 初始化分词器
        tokenizer = LogTokenizer()
        
        # 分词
        tokens = tokenizer.text_to_token_list(log_text)
        
        # 检测异常
        is_anomaly, score = is_log_anomaly(log_text, threshold)
        
        # 输出结果
        result = {
            "log_text": log_text,
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "tokens": tokens
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

def convert_log_file(file_path, output_path=None, anomaly_ratio=0.05, force=False):
    """转换日志文件为JSON格式"""
    if not LogConverter:
        print("错误: 日志转换功能不可用，缺少log_converter.py", file=sys.stderr)
        sys.exit(1)
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}", file=sys.stderr)
            sys.exit(1)
        
        print(f"开始转换日志文件: {file_path}")
        # 创建转换器
        converter = LogConverter(anomaly_ratio=anomaly_ratio)
        
        # 转换文件
        output_file = converter.convert_file(file_path, output_path, force)
        
        print(f"转换完成: {file_path} -> {output_file}")
        return output_file
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

def batch_detect_logs(file_path, threshold=None, output_path=None, is_json_format=False):
    """批量检测日志文件"""
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在: {file_path}", file=sys.stderr)
            sys.exit(1)
        
        # 如果不是指定的JSON格式，且不是自动检测到的JSON格式，则尝试普通日志处理
        if not is_json_format and not is_json_file(file_path):
            # 读取普通日志文件
            with open(file_path, 'r', encoding='utf-8') as f:
                logs = [line.strip() for line in f if line.strip()]
        else:
            # 读取JSON格式日志文件
            logs = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_data = json.loads(line.strip())
                        if isinstance(log_data, dict) and "text" in log_data:
                            logs.append(log_data["text"])
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # 如果解析失败，将整行作为日志文本
                        logs.append(line.strip())
        
        # 初始化分词器
        tokenizer = LogTokenizer()
        
        # 处理结果
        results = []
        anomaly_count = 0
        
        # 处理每条日志，添加进度条
        print(f"开始处理 {len(logs)} 条日志...")
        for log_text in tqdm(logs, desc="检测日志异常"):
            if not log_text:
                continue
            
            # 检测异常
            is_anomaly, score = is_log_anomaly(log_text, threshold)
            if is_anomaly:
                anomaly_count += 1
            
            # 分词
            tokens = tokenizer.text_to_token_list(log_text)
            
            # 添加到结果列表
            results.append({
                "log_text": log_text,
                "is_anomaly": is_anomaly,
                "anomaly_score": score,
                "tokens": tokens
            })
        
        # 计算统计数据
        total_logs = len(results)
        anomaly_percentage = (anomaly_count / total_logs * 100) if total_logs > 0 else 0
        
        # 准备输出结果
        output = {
            "results": results,
            "summary": {
                "total_logs": total_logs,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": anomaly_percentage
            }
        }
        
        # 输出结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_path}")
        else:
            print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
            print(f"检测到 {anomaly_count}/{total_logs} 条异常日志 ({anomaly_percentage:.2f}%)")
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

def tokenize_log(log_text):
    """将日志转换为token列表"""
    try:
        # 初始化分词器
        tokenizer = LogTokenizer()
        
        # 分词
        tokens = tokenizer.text_to_token_list(log_text)
        
        # 输出结果
        print(json.dumps({"tokens": tokens}, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"错误: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 