from app.app import app
from config import config

if __name__ == '__main__':

    # 加载配置
    app.config.from_object(config['development'])

    
    # 启动服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )