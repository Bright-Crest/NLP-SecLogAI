# 六、部署与维护

本章节详细描述NLP-SecLogAI系统的部署流程、环境配置和持续维护方案。

## 6.1 部署环境要求

### 6.1.1 硬件需求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 2核 | 4核或更高 |
| 内存 | 4GB | 8GB或更高 |
| 磁盘空间 | 20GB | 100GB或更高 |
| 网络 | 100Mbps | 1000Mbps |

### 6.1.2 软件需求

| 软件 | 版本 | 说明 |
|------|------|------|
| Python | 3.8+ | 运行环境 |
| PostgreSQL/MySQL | 10.0+/8.0+ | 数据库 |
| Docker | 20.10+ | 容器化部署(可选) |
| Redis | 6.0+ | 缓存和队列(可选) |
| Nginx | 1.18+ | 反向代理(生产环境) |

### 6.1.3 操作系统支持

- Ubuntu 20.04 LTS 或更高版本
- CentOS 8 或更高版本
- Debian 10 或更高版本
- Windows Server 2019 或更高版本(需额外配置)

## 6.2 安装与配置

### 6.2.1 基础环境配置

#### Linux环境配置

```bash
# 更新系统包
sudo apt update && sudo apt upgrade -y   # Ubuntu/Debian
# 或
sudo yum update -y   # CentOS/RHEL

# 安装Python和相关工具
sudo apt install -y python3 python3-pip python3-venv   # Ubuntu/Debian
# 或
sudo yum install -y python3 python3-pip python3-devel   # CentOS/RHEL

# 安装数据库
sudo apt install -y postgresql postgresql-contrib   # Ubuntu/Debian
# 或
sudo yum install -y postgresql postgresql-server   # CentOS/RHEL

# 安装其他依赖
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev   # Ubuntu/Debian
# 或
sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel   # CentOS/RHEL
```

#### 数据库初始化

```bash
# PostgreSQL初始化
sudo -u postgres psql

postgres=# CREATE DATABASE seclogai;
postgres=# CREATE USER seclogai_user WITH PASSWORD 'your_password';
postgres=# GRANT ALL PRIVILEGES ON DATABASE seclogai TO seclogai_user;
postgres=# \q
```

### 6.2.2 应用部署

#### 方法一：直接部署

```bash
# 克隆代码仓库
git clone https://github.com/yourusername/NLP-SecLogAI.git
cd NLP-SecLogAI

# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或
venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，配置数据库连接等信息

# 初始化数据库
flask db upgrade

# 启动应用
python run.py
```

#### 方法二：Docker部署

```bash
# 克隆代码仓库
git clone https://github.com/yourusername/NLP-SecLogAI.git
cd NLP-SecLogAI

# 构建Docker镜像
docker build -t nlp-seclogai .

# 使用docker-compose启动服务
docker-compose up -d
```

docker-compose.yml配置示例：
```yaml
version: '3.8'

services:
  web:
    build: .
    restart: always
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://seclogai_user:your_password@db:5432/seclogai
      - SECRET_KEY=your_secret_key
      - FLASK_ENV=production
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads

  db:
    image: postgres:14
    restart: always
    environment:
      - POSTGRES_USER=seclogai_user
      - POSTGRES_PASSWORD=your_password
      - POSTGRES_DB=seclogai
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6
    restart: always
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 6.2.3 生产环境配置

#### Nginx配置

创建Nginx配置文件`/etc/nginx/sites-available/seclogai`：

```nginx
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/NLP-SecLogAI/static;
        expires 30d;
    }
}
```

启用配置并重启Nginx：

```bash
sudo ln -s /etc/nginx/sites-available/seclogai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Supervisor配置

创建Supervisor配置文件`/etc/supervisor/conf.d/seclogai.conf`：

```ini
[program:seclogai]
directory=/path/to/NLP-SecLogAI
command=/path/to/NLP-SecLogAI/venv/bin/gunicorn -b 127.0.0.1:5000 -w 4 --timeout 120 app:app
user=www-data
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/seclogai/seclogai.err.log
stdout_logfile=/var/log/seclogai/seclogai.out.log
```

重新加载Supervisor配置：

```bash
sudo mkdir -p /var/log/seclogai
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start seclogai
```

## 6.3 系统升级

### 6.3.1 常规升级流程

```bash
# 进入项目目录
cd /path/to/NLP-SecLogAI

# 拉取最新代码
git pull

# 激活虚拟环境
source venv/bin/activate

# 更新依赖
pip install -r requirements.txt

# 执行数据库迁移
flask db upgrade

# 重启应用
sudo supervisorctl restart seclogai
```

### 6.3.2 Docker环境升级

```bash
# 进入项目目录
cd /path/to/NLP-SecLogAI

# 拉取最新代码
git pull

# 重新构建镜像
docker-compose build

# 重启服务
docker-compose down
docker-compose up -d
```

### 6.3.3 升级回滚策略

```bash
# 查看git提交历史
git log --oneline

# 回滚到特定版本
git checkout <commit-hash>

# 激活虚拟环境并回滚数据库
source venv/bin/activate
flask db downgrade

# 重启应用
sudo supervisorctl restart seclogai
```

## 6.4 备份与恢复

### 6.4.1 数据库备份

#### PostgreSQL备份

```bash
# 创建备份目录
mkdir -p /backup/seclogai

# 执行完整备份
pg_dump -U seclogai_user -h localhost -d seclogai > /backup/seclogai/seclogai_$(date +%Y%m%d).sql

# 自动化备份脚本
cat > /etc/cron.daily/backup-seclogai <<EOF
#!/bin/bash
BACKUP_DIR="/backup/seclogai"
TIMESTAMP=\$(date +%Y%m%d)
pg_dump -U seclogai_user -h localhost -d seclogai > \$BACKUP_DIR/seclogai_\$TIMESTAMP.sql
find \$BACKUP_DIR -name "seclogai_*.sql" -mtime +30 -delete
EOF

chmod +x /etc/cron.daily/backup-seclogai
```

#### 应用数据备份

```bash
# 备份配置和上传文件
tar -czf /backup/seclogai/app_data_$(date +%Y%m%d).tar.gz /path/to/NLP-SecLogAI/.env /path/to/NLP-SecLogAI/uploads /path/to/NLP-SecLogAI/instance
```

### 6.4.2 数据恢复

#### PostgreSQL恢复

```bash
# 恢复数据库
psql -U seclogai_user -h localhost -d seclogai < /backup/seclogai/seclogai_YYYYMMDD.sql
```

#### 应用数据恢复

```bash
# 恢复配置和上传文件
tar -xzf /backup/seclogai/app_data_YYYYMMDD.tar.gz -C /
```

## 6.5 监控与日志管理

### 6.5.1 系统监控

#### Prometheus配置

创建Prometheus配置文件`/etc/prometheus/prometheus.yml`：

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'seclogai'
    static_configs:
      - targets: ['localhost:5000']
```

#### Grafana仪表盘设置

1. 安装Grafana
2. 添加Prometheus数据源
3. 导入预设仪表盘或创建自定义仪表盘

### 6.5.2 日志管理

#### 日志轮转配置

创建logrotate配置文件`/etc/logrotate.d/seclogai`：

```
/var/log/seclogai/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        supervisorctl restart seclogai
    endscript
}
```

#### 集中式日志收集(可选)

使用ELK(Elasticsearch, Logstash, Kibana)或Graylog进行集中式日志收集。

## 6.6 性能优化

### 6.6.1 应用性能优化

1. **数据库优化**
   - 定期VACUUM和ANALYZE操作
   - 创建适当的索引
   - 优化查询语句

2. **Web服务器优化**
   - 增加worker进程数量
   - 使用异步任务处理长时间运行的操作
   - 启用缓存

3. **NLP处理优化**
   - 模型量化
   - 批处理请求
   - 分布式计算

### 6.6.2 扩展性方案

#### 水平扩展

```
用户请求 -> 负载均衡器 -> [Web服务器集群] -> [数据库集群]
                          |
                          -> [NLP处理集群]
```

#### 垂直扩展

- 增加服务器CPU和内存
- 使用更快的存储解决方案
- 优化代码和算法

## 6.7 安全加固

### 6.7.1 应用安全配置

1. **Web安全**
   - 启用HTTPS
   - 设置适当的CSP(Content Security Policy)
   - 实现CSRF保护
   - 设置安全Cookie属性

2. **认证与授权**
   - 实现多因素认证
   - 角色基础访问控制
   - 会话超时设置

3. **数据安全**
   - 敏感数据加密存储
   - 数据库访问控制
   - 备份加密

### 6.7.2 服务器安全加固

1. **操作系统加固**
   - 定期更新补丁
   - 最小权限原则
   - 禁用不必要的服务
   - 配置防火墙

2. **容器安全**
   - 使用最小化基础镜像
   - 非root用户运行
   - 定期更新容器镜像
   - 扫描容器漏洞

## 6.8 故障排除

### 6.8.1 常见问题及解决方案

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 应用无法启动 | 依赖问题、配置错误 | 检查错误日志、验证依赖安装、检查配置文件 |
| 数据库连接失败 | 凭证错误、网络问题 | 验证数据库凭证、检查网络连接、查看数据库状态 |
| NLP处理缓慢 | 资源不足、模型过大 | 增加资源、优化模型、使用批处理 |
| 内存溢出 | 数据量过大、内存泄漏 | 增加内存、优化代码、限制批处理大小 |
| 磁盘空间不足 | 日志过多、临时文件积累 | 清理日志、配置日志轮转、删除临时文件 |

### 6.8.2 性能问题诊断

1. **CPU使用率高**
   - 使用`top`、`htop`识别高CPU进程
   - 检查应用日志中的耗时操作
   - 分析NLP处理或复杂查询

2. **内存使用率高**
   - 使用`free`、`vmstat`监控内存
   - 检查Python进程内存使用
   - 考虑内存泄漏可能性

3. **数据库性能问题**
   - 使用`EXPLAIN`分析查询性能
   - 检查慢查询日志
   - 优化索引和查询语句

### 6.8.3 日志分析工具

- `journalctl`：系统日志查询
- `tail -f`：实时查看日志
- `grep`：搜索特定错误消息
- ELK/Graylog：高级日志分析 