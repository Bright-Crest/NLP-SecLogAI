#!/usr/bin/env python3
"""
日志转换工具
将普通日志文件转换为适合train.py训练的JSON格式文件
适用于HDFS等常见日志格式
"""

import re
import os
import json
import argparse
import random
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.log_tokenizer import LogTokenizer

class LogConverter:
    """
    日志转换器
    将普通文本日志转换为带有标签的JSON格式，适合模型训练
    """
    
    def __init__(self, anomaly_ratio=0.05, random_seed=42):
        """
        初始化转换器
        params:
            anomaly_ratio: 在没有标签的情况下使用的异常比例（模拟异常）
            random_seed: 随机种子，确保结果可复现
        """
        self.tokenizer = LogTokenizer()
        self.anomaly_ratio = anomaly_ratio
        self.random_seed = random_seed
        random.seed(random_seed)
    
    def is_json_file(self, file_path):
        """
        检查文件是否为JSON格式
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 尝试读取第一行并解析为JSON
                line = f.readline().strip()
                json.loads(line)
                return True
        except (json.JSONDecodeError, UnicodeDecodeError, IOError):
            return False
        
    def generate_output_path(self, input_path):
        """
        生成输出文件路径
        格式: 原文件路径_jsonl.json
        """
        dir_name = os.path.dirname(input_path) or '.'
        base_name = os.path.basename(input_path)
        file_name, _ = os.path.splitext(base_name)
        return os.path.join(dir_name, f"{file_name}_jsonl.json")
    
    def check_existing_file(self, output_path):
        """
        检查输出文件是否已存在
        """
        if os.path.exists(output_path):
            # 检查文件是否为有效的JSON格式
            if self.is_json_file(output_path):
                return True
        return False
    
    def convert_file(self, input_path, output_path=None, force=False):
        """
        转换日志文件
        params:
            input_path: 输入文件路径
            output_path: 输出文件路径，默认为自动生成
            force: 是否强制转换，即使输出文件已存在
        return:
            输出文件路径
        """
        # 确定输出路径
        if output_path is None:
            output_path = self.generate_output_path(input_path)
        
        # 检查输出文件是否已存在
        if not force and self.check_existing_file(output_path):
            print(f"输出文件已存在: {output_path}，跳过转换")
            return output_path
        
        # 读取输入文件
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            # 尝试其他编码方式
            encodings = ['latin-1', 'gbk', 'utf-16']
            for encoding in encodings:
                try:
                    with open(input_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise Exception(f"无法解码文件: {input_path}")
        
        # 检查是否已经是JSON格式
        if len(lines) > 0:
            try:
                json.loads(lines[0].strip())
                # 如果能成功解析为JSON，检查是否包含必要字段
                data = json.loads(lines[0].strip())
                if isinstance(data, dict) and "text" in data and "label" in data:
                    print(f"输入文件已经是适合的JSON格式，直接复制到: {output_path}")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for line in lines:
                            f.write(line)
                    return output_path
            except (json.JSONDecodeError, TypeError):
                # 不是JSON格式，需要转换
                pass
        
        # 转换日志文件
        converted_logs = []
        for line in tqdm(lines, desc="转换日志"):
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            # 预处理日志，去除时间戳等
            tokens = self.tokenizer.text_to_token_list(line)
            if not tokens:  # 跳过无有效内容的日志
                continue
            
            # 模拟异常标签（随机分配）
            label = 1 if random.random() < self.anomaly_ratio else 0
            
            # 创建JSON记录
            log_entry = {
                "text": line,
                "tokens": tokens,
                "label": label
            }
            converted_logs.append(log_entry)
        
        # 写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for log_entry in converted_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        print(f"转换完成，共处理 {len(converted_logs)} 条日志，已保存到: {output_path}")
        return output_path
    
    def analyze_logs(self, input_path):
        """
        分析日志
        - 计算日志长度统计
        - 识别常见模式
        - 确定可能的异常
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(input_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except:
                raise Exception(f"无法读取文件: {input_path}")
        
        # 删除空行
        lines = [line.strip() for line in lines if line.strip()]
        
        # 分析日志长度
        log_lengths = [len(line) for line in lines]
        avg_length = sum(log_lengths) / len(log_lengths) if log_lengths else 0
        
        # 分析日志模式（简单方法：检查常见前缀）
        prefixes = {}
        for line in lines[:1000]:  # 仅分析前1000行
            # 提取前20个字符作为前缀
            prefix = line[:20].strip()
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        # 获取前5个最常见前缀
        common_prefixes = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 生成分析结果
        analysis = {
            "total_logs": len(lines),
            "average_length": avg_length,
            "common_prefixes": [prefix for prefix, count in common_prefixes],
            "prefix_counts": [count for prefix, count in common_prefixes],
            "suggested_anomaly_ratio": min(0.05, 10 / len(lines))  # 建议的异常比例
        }
        
        return analysis

def main():
    parser = argparse.ArgumentParser(description="日志转换工具 - 将普通日志转换为训练所需的JSON格式")
    
    # 必需参数
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="输入日志文件路径")
    
    # 可选参数
    parser.add_argument("--output", "-o", type=str, default=None, 
                        help="输出JSON文件路径，默认为输入文件路径+'_jsonl.json'")
    parser.add_argument("--anomaly-ratio", "-r", type=float, default=0.05, 
                        help="模拟的异常比例，默认为0.05")
    parser.add_argument("--force", "-f", action="store_true", 
                        help="强制转换，即使输出文件已存在")
    parser.add_argument("--analyze", "-a", action="store_true", 
                        help="仅分析日志，不进行转换")
    parser.add_argument("--seed", "-s", type=int, default=42, 
                        help="随机种子，确保结果可复现")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 创建转换器
    converter = LogConverter(anomaly_ratio=args.anomaly_ratio, random_seed=args.seed)
    
    # 仅分析模式
    if args.analyze:
        print(f"分析日志文件: {args.input}")
        analysis = converter.analyze_logs(args.input)
        print("\n=== 日志分析结果 ===")
        print(f"总日志数: {analysis['total_logs']}")
        print(f"平均日志长度: {analysis['average_length']:.2f} 字符")
        print("\n常见日志前缀:")
        for i, (prefix, count) in enumerate(zip(analysis['common_prefixes'], analysis['prefix_counts'])):
            print(f"{i+1}. '{prefix}...' ({count} 条)")
        
        print(f"\n建议的异常比例: {analysis['suggested_anomaly_ratio']:.4f}")
        return 0
    
    # 转换模式
    output_path = converter.convert_file(args.input, args.output, args.force)
    print(f"转换完成: {args.input} -> {output_path}")
    return 0

if __name__ == "__main__":
    exit(main()) 