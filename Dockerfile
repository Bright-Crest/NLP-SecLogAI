# 使用官方Python镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，防止Python生成.pyc文件并确保日志直接输出到控制台
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统级依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 首先复制并安装项目主要依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制并安装AI检测模块的额外依赖
COPY ai_detect/requirements.txt ./ai_detect-requirements.txt
RUN pip install --no-cache-dir -r ai_detect-requirements.txt

# 复制.env文件（如果存在）
COPY .env* ./

# 复制项目文件
COPY . .

# 创建必要的目录（如果不存在）
RUN mkdir -p instance

# 设置端口
EXPOSE 5000

# 启动应用
CMD ["python", "run.py"] 