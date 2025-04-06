# 测试文档

本目录包含 NLP-SecLogAI 的测试代码，用于确保各个模块的正确功能。

## 目录结构

```
tests/
├── unit/                      # 单元测试目录
│   ├── test_nlp_processor.py  # NLP处理器模块的测试
├── __init__.py                # 测试包初始化文件
└── README.md                  # 本文档
```

## 运行测试

运行以下命令执行所有测试：

```bash
# 在项目根目录下
cd log_analysis_project
python -m unittest discover tests

# 运行特定测试
python -m unittest tests.unit.test_nlp_processor
```

## 测试内容

### NLP处理器测试

`test_nlp_processor.py` 文件测试了 NLP 处理器的以下功能：

1. **查询预处理功能**：测试时间表达式、用户名、IP地址的识别和处理
2. **时间处理功能**：测试相对时间和特殊时间的解析与转换
3. **关键词映射功能**：测试状态关键词如"失败"、"成功"的映射
4. **模式匹配回退功能**：测试在API不可用时的回退生成SQL能力
5. **SQL提取功能**：测试从模型输出中提取SQL语句的能力

## 添加新测试

添加新测试时，请遵循以下规则：

1. 为每个测试方法编写清晰的文档字符串，说明测试目的
2. 使用合适的断言验证结果
3. 避免依赖外部服务，如OpenRouter API（除非是集成测试）
4. 保持测试的独立性，每个测试不应该依赖其他测试的结果 