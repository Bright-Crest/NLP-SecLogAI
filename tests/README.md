# NLP-SecLogAI 测试框架

本文档介绍了NLP-SecLogAI项目的测试框架、结构和运行方法。

## 测试结构

测试代码组织结构如下：

```
tests/
├── ai_detect/              # AI检测模块测试
│   ├── unit/               # 单元测试
│   └── integration/        # 集成测试
├── app/                    # 应用模块测试
│   ├── unit/               # 单元测试
│   ├── integration/        # 集成测试
│   ├── functional/         # 功能测试
│   └── performance/        # 性能测试
├── run_tests.py            # 测试运行器
└── README.md               # 测试文档
```

## 测试类型

项目包含以下几种类型的测试：

1. **单元测试**：测试独立组件的功能
   - 测试AI检测CLI功能
   - 测试训练功能
   - 测试日志标记化
   - 测试异常评分服务

2. **集成测试**：测试组件之间的交互
   - 测试AI异常检测流程
   - 测试API路由与服务的集成

3. **功能测试**：测试完整功能
   - 测试API端点功能

4. **性能测试**：测试系统性能
   - 测试API在不同负载下的响应时间
   - 测试并发处理能力
   - 测试系统稳定性

## 运行测试

### 依赖安装

首先安装测试依赖：

```bash
pip install -r requirements.txt
```

### 使用测试运行器

可以使用测试运行器 `run_tests.py` 来运行测试：

```bash
# 运行所有测试
python tests/run_tests.py

# 运行特定类型的测试
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type functional
python tests/run_tests.py --type performance

# 运行特定模块的测试
python tests/run_tests.py --type ai_detect
python tests/run_tests.py --type app

# 使用模式匹配特定测试文件
python tests/run_tests.py --pattern "test_anomaly*.py"

# 设置输出详细度
python tests/run_tests.py --verbosity 2  # 0=安静, 1=进度, 2=详细(默认)
```

### 使用pytest

也可以直接使用pytest运行测试：

```bash
# 运行所有测试
pytest tests/

# 运行特定目录的测试
pytest tests/ai_detect/
pytest tests/app/unit/

# 运行特定文件的测试
pytest tests/ai_detect/unit/test_ai_detect_cli.py

# 生成测试覆盖率报告
pytest --cov=ai_detect --cov=app tests/
```

## 测试覆盖率

运行以下命令生成详细的测试覆盖率报告：

```bash
coverage run -m pytest tests/
coverage report -m
coverage html  # 生成HTML报告
```

HTML报告将生成在 `htmlcov/` 目录下，可以用浏览器打开 `htmlcov/index.html` 查看。

## 持续集成

在CI/CD流程中，可以使用以下命令运行测试：

```bash
python tests/run_tests.py --type unit --type integration
```

性能测试通常不在CI流程中运行，可以在特定环境下单独运行：

```bash
python tests/run_tests.py --type performance
``` 
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
