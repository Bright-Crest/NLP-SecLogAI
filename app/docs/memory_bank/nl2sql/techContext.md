# NL2SQL技术上下文

## 技术栈

NL2SQL模块使用以下技术栈：

| 组件 | 技术 | 版本 | 说明 |
|------|------|------|------|
| 后端框架 | Flask | 2.0+ | Web服务框架 |
| 数据库 | SQLite | 3.0+ | 轻量级关系型数据库 |
| 大语言模型 | Claude/GPT通过OpenRouter | - | 自然语言处理与SQL生成 |
| API调用 | requests | 2.25+ | HTTP请求库 |
| 正则处理 | re (Python标准库) | - | 模式匹配与文本处理 |
| 日期处理 | datetime (Python标准库) | - | 时间表达式处理 |
| 环境变量 | python-dotenv | 0.19+ | 环境配置管理 |
| 前端框架 | Bootstrap | 5.1+ | CSS框架 |
| 前端交互 | jQuery | 3.6+ | JavaScript库 |

## 开发环境

### 环境配置

1. **Python 环境**：
   - Python 3.9+
   - 虚拟环境管理（推荐使用venv或conda）

2. **依赖管理**：
   - 使用requirements.txt管理Python包依赖
   - 主要依赖：Flask, requests, python-dotenv

3. **API密钥**：
   - 需要在.env文件中配置OPENROUTER_API_KEY
   - 格式: `OPENROUTER_API_KEY=your_api_key_here`

4. **开发工具**：
   - 推荐使用VSCode或PyCharm
   - 安装Python、Flask和SQLite相关插件

### 本地开发流程

1. 克隆项目并进入项目目录
   ```bash
   git clone <project-url>
   cd NLP-SecLogAI
   ```

2. 创建并激活虚拟环境
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

4. 配置环境变量
   ```bash
   cp .env.example .env
   # 编辑.env文件，添加OPENROUTER_API_KEY
   ```

5. 运行开发服务器
   ```bash
   python run.py
   ```

6. 访问NL2SQL界面
   ```
   http://localhost:5000/nlp/ui
   ```

## 技术限制

### API依赖限制

1. **OpenRouter依赖**：
   - 主要功能依赖OpenRouter API
   - 无API环境下降级为模式匹配方案，功能有限
   - API调用可能存在延迟（通常1-3秒）

2. **模型限制**：
   - 依赖LLM的SQL生成质量
   - 可能存在幻觉问题，生成不符合预期的SQL
   - 处理非常复杂的查询能力有限

### 性能考量

1. **查询处理时间**：
   - API调用通常需要1-3秒
   - 复杂查询执行可能需要额外时间
   - 整体响应时间目标控制在4秒以内

2. **并发处理**：
   - 当前实现不支持高并发场景
   - 未实现查询缓存机制
   - API调用存在速率限制

### 安全考量

1. **SQL注入防护**：
   - 系统生成SQL，而非直接执行用户输入
   - 使用参数化查询执行最终SQL
   - 但LLM生成的SQL仍需谨慎处理

2. **权限控制**：
   - 当前版本无细粒度权限控制
   - 所有查询均使用相同数据库权限
   - 未实现查询结果脱敏处理

## 依赖关系

### 内部依赖

NL2SQL模块依赖系统内的以下组件：

1. **数据库模块**：
   - 依赖`models/db.py`中的数据库连接
   - 使用`TABLE_SCHEMA`获取表结构信息

2. **Web框架**：
   - 依赖Flask应用上下文
   - 使用Blueprint注册路由

3. **前端资源**：
   - 使用`templates/nlp_sql.html`页面模板
   - 依赖Bootstrap和jQuery资源

### 外部依赖

主要外部依赖包括：

1. **OpenRouter API**：
   - 用于自然语言处理和SQL生成
   - 需要有效的API密钥
   - 接口文档：https://openrouter.ai/docs

2. **Python包**：
   - requests：处理HTTP请求
   - python-dotenv：环境变量管理
   - Flask：Web框架
   - re, json, datetime：标准库

## 版本兼容性

当前NL2SQL模块经过测试，与以下环境兼容：

1. **Python版本**：3.7, 3.8, 3.9, 3.10
2. **操作系统**：Windows 10/11, Ubuntu 20.04+, macOS 11.0+
3. **浏览器**：Chrome 90+, Firefox 88+, Edge 90+, Safari 14+

## 部署注意事项

1. **环境变量**：
   - 生产环境必须配置OPENROUTER_API_KEY
   - 可选配置MODEL_NAME来指定特定模型

2. **数据库**：
   - 确保数据库模式符合预期
   - 表结构变更需同步更新TABLE_SCHEMA

3. **API密钥管理**：
   - 生产环境API密钥应安全管理
   - 考虑使用密钥轮换策略
   - 监控API使用量和配额 