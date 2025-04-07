FROM python:3.11-slim

ARG USE_GPU=true
ARG CUDA=118

# 设置环境变量，防止Python生成.pyc文件并确保日志直接输出到控制台
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /NLP_SecLogAI
WORKDIR /NLP_SecLogAI

# 设置 pip 源为清华镜像
RUN mkdir -p /etc/pip \
    && echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > /etc/pip.conf

RUN pip install --no-cache-dir --upgrade pip

RUN if [ "$USE_GPU" = "false" ]; then \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    else \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu${CUDA}; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_detect/requirements.txt ai_detect/requirements.txt
RUN pip install --no-cache-dir -r ai_detect/requirements.txt

COPY tests/requirements.txt tests/requirements.txt
RUN pip install --no-cache-dir -r tests/requirements.txt

COPY app/ app/
COPY geolite/ geolite/
COPY config.py config.py
COPY run.py run.py

COPY ai_detect/ ai_detect/

COPY tests/ tests/

EXPOSE 5000

CMD ["python", "run.py"]
