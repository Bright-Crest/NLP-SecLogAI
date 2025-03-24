FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY ai/requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制AI模块代码
COPY ai/ .

# 创建必要的目录
RUN mkdir -p training/checkpoints \
    training/logs \
    training/models

# 设置环境变量
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/training/models

# 暴露API端口
EXPOSE 8001

# 启动API服务
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8001"] 