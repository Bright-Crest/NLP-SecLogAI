#!/usr/bin/env python3
"""
完整工作流程示例脚本
展示从原始日志到模型训练、评估、对比的完整流程
"""

import os
import subprocess
import argparse
import json
import time

def run_command(command, description=None):
    """执行命令并打印输出"""
    if description:
        print(f"\n=== {description} ===")
    
    print(f"执行命令: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.stderr:
        print(f"错误: {result.stderr}")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="TinyLogBERT工作流程示例")
    
    # 日志文件参数
    parser.add_argument("--raw-logs", type=str, required=True, 
                      help="原始日志文件路径")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, 
                      help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, 
                      help="批大小")
    parser.add_argument("--sample-limit", type=int, default=10000, 
                      help="样本数量限制")
    
    # 输出目录
    parser.add_argument("--output-dir", type=str, default="./workflow_output", 
                      help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    
    print(f"TinyLogBERT工作流程示例 - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"原始日志文件: {args.raw_logs}")
    print(f"输出目录: {args.output_dir}")
    
    # 步骤1：分析日志文件
    run_command(
        f"python log_converter.py --input {args.raw_logs} --analyze",
        "步骤1: 分析日志文件"
    )
    
    # 步骤2：转换日志文件
    json_logs = os.path.join(args.output_dir, "converted_logs.json")
    run_command(
        f"python log_converter.py --input {args.raw_logs} --output {json_logs} --force",
        "步骤2: 转换日志文件"
    )
    
    # 步骤3：划分训练集和测试集
    split_ratio = 0.8
    with open(json_logs, 'r', encoding='utf-8') as f:
        logs = [line.strip() for line in f if line.strip()]
    
    train_size = int(len(logs) * split_ratio)
    train_logs = logs[:train_size]
    test_logs = logs[train_size:]
    
    train_file = os.path.join(args.output_dir, "train_logs.json")
    test_file = os.path.join(args.output_dir, "test_logs.json")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_logs:
            f.write(line + '\n')
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for line in test_logs:
            f.write(line + '\n')
    
    print(f"\n=== 步骤3: 划分数据集 ===")
    print(f"总日志数: {len(logs)}")
    print(f"训练集: {len(train_logs)} 条")
    print(f"测试集: {len(test_logs)} 条")
    
    # 步骤4：训练TinyLogBERT模型
    tiny_model_path = os.path.join(checkpoint_dir, "tiny_model.pt")
    run_command(
        f"python train.py --data-file {train_file} --model-type tinylogbert "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --sample-limit {args.sample_limit} "
        f"--checkpoint-dir {checkpoint_dir}",
        "步骤4: 训练TinyLogBERT模型"
    )
    
    # 步骤5：训练LogBERT-Distil模型
    distil_model_path = os.path.join(checkpoint_dir, "distil_model.pt")
    run_command(
        f"python train.py --data-file {train_file} --model-type logbert_distil "
        f"--epochs {args.epochs} --batch-size {args.batch_size} --sample-limit {args.sample_limit} "
        f"--checkpoint-dir {checkpoint_dir}",
        "步骤5: 训练LogBERT-Distil模型"
    )
    
    # 步骤6：模型性能比较
    run_command(
        f"python model_comparison_example.py --test-data {test_file} "
        f"--tiny-model-path {tiny_model_path} --distil-model-path {distil_model_path} "
        f"--batch-size {args.batch_size} --embedding-samples 200",
        "步骤6: 模型性能比较"
    )
    
    # 步骤7：使用CLI工具进行异常检测
    results_file = os.path.join(args.output_dir, "detection_results.json")
    run_command(
        f"python ai_detect_cli.py batch {test_file} --output {results_file} --json-format",
        "步骤7: 使用CLI工具进行异常检测"
    )
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n=== 工作流程完成 ===")
    print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    print(f"结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 