#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单元测试运行器
用于运行app目录下的单元测试
"""

import unittest
import sys
import os
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

def run_tests(test_pattern=None, verbose=1, failfast=False):
    """
    运行单元测试
    
    参数:
        test_pattern: 测试模式匹配字符串，例如"test_log_tokenizer"
        verbose: 详细程度，0-3
        failfast: 是否在第一个失败时停止
    
    返回:
        测试结果
    """
    # 创建测试加载器
    loader = unittest.TestLoader()
    
    # 确定测试目录
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 根据模式选择测试
    if test_pattern:
        # 加载匹配名称的测试
        suite = loader.discover(test_dir, pattern=f"*{test_pattern}*.py")
    else:
        # 加载所有测试
        suite = loader.discover(test_dir)
    
    # 创建测试运行器
    runner = unittest.TextTestRunner(verbosity=verbose, failfast=failfast)
    
    # 运行测试
    print(f"正在运行测试... 目录: {test_dir}")
    result = runner.run(suite)
    
    # 打印结果摘要
    print("\n测试结果摘要:")
    print(f"运行测试: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    return result


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='运行单元测试')
    parser.add_argument('-p', '--pattern', type=str, default=None,
                      help='测试模式匹配字符串，例如"test_log_tokenizer"')
    parser.add_argument('-v', '--verbose', type=int, default=2, choices=[0, 1, 2, 3],
                      help='详细程度，0-3')
    parser.add_argument('-f', '--failfast', action='store_true',
                      help='在第一个测试失败时停止')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 运行测试
    result = run_tests(args.pattern, args.verbose, args.failfast)
    
    # 根据测试结果设置退出代码
    sys.exit(len(result.failures) + len(result.errors))