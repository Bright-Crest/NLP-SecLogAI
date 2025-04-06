#!/usr/bin/env python
import os
import sys
import unittest
import argparse
import time

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(ROOT_DIR, 'tests')
sys.path.append(TEST_DIR)


def discover_tests(test_type=None):
    """发现测试用例"""
    if test_type is None:
        # 发现所有测试
        start_dir = os.path.join(ROOT_DIR, 'tests')
    else:
        # 发现特定类型测试
        if test_type == 'unit':
            patterns = ['tests/*/unit', 'tests/unit']
        elif test_type == 'integration':
            patterns = ['tests/*/integration']
        elif test_type == 'functional':
            patterns = ['tests/*/functional']
        elif test_type == 'performance':
            patterns = ['tests/*/performance']
        elif test_type == 'ai_detect':
            patterns = ['tests/ai_detect']
        elif test_type == 'app':
            patterns = ['tests/app']
        else:
            print(f"未知的测试类型: {test_type}")
            patterns = []
        
        # 使用指定的模式发现测试
        suite = unittest.TestSuite()
        for pattern in patterns:
            start_dir = os.path.join(ROOT_DIR, pattern)
            if os.path.exists(start_dir):
                sub_suite = unittest.defaultTestLoader.discover(start_dir, pattern='test_*.py')
                suite.addTest(sub_suite)
        return suite
    
    # 使用默认发现机制
    return unittest.defaultTestLoader.discover(start_dir, pattern='test_*.py', top_level_dir=ROOT_DIR)


def run_tests(test_suite, verbosity=2):
    """运行测试用例"""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # 打印测试总结
    print("\n=== 测试执行总结 ===")
    print(f"运行测试: {result.testsRun}")
    print(f"通过测试: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败测试: {len(result.failures)}")
    print(f"错误测试: {len(result.errors)}")
    print(f"测试执行时间: {end_time - start_time:.2f} 秒")
    
    # 打印失败和错误的测试详情
    if result.failures:
        print("\n=== 失败测试详情 ===")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{i}. {test}")
            print("-" * 80)
            print(traceback)
    
    if result.errors:
        print("\n=== 错误测试详情 ===")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{i}. {test}")
            print("-" * 80)
            print(traceback)
    
    # 返回是否全部通过
    return len(result.failures) == 0 and len(result.errors) == 0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行NLP-SecLogAI项目测试")
    parser.add_argument('--type', choices=['unit', 'integration', 'functional', 'performance', 'ai_detect', 'app'], 
                      help='要运行的测试类型')
    parser.add_argument('--verbosity', type=int, default=2, choices=[0, 1, 2],
                      help='输出详细级别 (0=安静, 1=进度, 2=详细)')
    parser.add_argument('--pattern', type=str, help='测试文件匹配模式，如"test_anomaly*.py"')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 发现测试
    if args.pattern:
        # 使用自定义模式发现测试
        if args.type:
            if args.type == 'unit':
                start_dirs = ['tests/*/unit', 'tests/unit']
            elif args.type == 'integration':
                start_dirs = ['tests/*/integration']
            elif args.type == 'functional':
                start_dirs = ['tests/*/functional']
            elif args.type == 'performance':
                start_dirs = ['tests/*/performance']
            elif args.type == 'ai_detect':
                start_dirs = ['tests/ai_detect']
            elif args.type == 'app':
                start_dirs = ['tests/app']
            else:
                start_dirs = ['tests']
        else:
            start_dirs = ['tests']
        
        # 使用指定的模式发现测试
        suite = unittest.TestSuite()
        for start_dir_pattern in start_dirs:
            for start_dir in [p for p in [os.path.join(ROOT_DIR, d) for d in [start_dir_pattern]] if os.path.exists(p)]:
                sub_suite = unittest.defaultTestLoader.discover(start_dir, pattern=args.pattern, top_level_dir=ROOT_DIR)
                suite.addTest(sub_suite)
    else:
        # 使用测试类型发现测试
        suite = discover_tests(args.type)
    
    # 运行测试
    success = run_tests(suite, args.verbosity)
    
    # 返回退出代码
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main()) 