#!/usr/bin/env python3
"""
模型比较示例脚本
展示如何使用模型比较服务比较TinyLogBERT和LogBERT-Distil的性能
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.services.model_comparison import compare_models, compare_model_embeddings

def load_test_data(data_file):
    """加载测试数据"""
    print(f"加载测试数据: {data_file}")
    test_data = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log = json.loads(line.strip())
                test_data.append(log)
            except json.JSONDecodeError as e:
                print(f"解析行出错: {line} - {str(e)}")
    
    print(f"共加载 {len(test_data)} 条测试数据")
    return test_data

def main():
    parser = argparse.ArgumentParser(description="模型性能比较工具")
    
    # 必需参数
    parser.add_argument("--test-data", type=str, required=True, help="测试数据文件路径（每行一个JSON）")
    
    # 模型配置
    parser.add_argument("--tiny-model-path", type=str, default="./checkpoint/tiny_model.pt", 
                        help="TinyLogBERT模型路径")
    parser.add_argument("--distil-model-path", type=str, default="./checkpoint/distil_model.pt", 
                        help="LogBERT-Distil模型路径")
    
    # 其他参数
    parser.add_argument("--batch-size", type=int, default=32, help="测试批大小")
    parser.add_argument("--embedding-samples", type=int, default=200, 
                        help="嵌入向量比较的样本数量")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.test_data):
        print(f"错误: 测试数据文件不存在: {args.test_data}")
        return 1
    
    # 加载测试数据
    test_data = load_test_data(args.test_data)
    if not test_data:
        print("错误: 没有加载到测试数据")
        return 1
    
    # 模型配置
    model_configs = [
        {
            "name": "TinyLogBERT",
            "path": args.tiny_model_path,
            "type": "tinylogbert"
        },
        {
            "name": "LogBERT-Distil",
            "path": args.distil_model_path,
            "type": "logbert_distil"
        }
    ]
    
    print("\n=== 开始模型性能比较 ===")
    # 比较模型性能
    results = compare_models(test_data, model_configs, batch_size=args.batch_size)
    
    # 打印结果摘要
    print("\n=== 模型性能对比 ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    
    print("\n=== 开始嵌入向量比较 ===")
    # 比较嵌入向量
    compare_model_embeddings(test_data, model_configs, sample_size=args.embedding_samples)
    
    print("\n比较完成！结果保存在 comparison_results 目录中。")
    return 0

if __name__ == "__main__":
    exit(main()) 