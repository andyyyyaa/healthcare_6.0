#!/usr/bin/env python3
"""
生产环境启动脚本
解决flash失效和登录状态问题
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # 设置环境变量
    os.environ['FLASK_ENV'] = 'production'
    
    # 获取配置
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    
    print("=== 生产环境启动配置 ===")
    print(f"主机: {host}")
    print(f"端口: {port}")
    print(f"Flask环境: {os.environ.get('FLASK_ENV')}")
    print(f"Secret Key: {app.config.get('SECRET_KEY', 'Not Set')[:20]}...")
    print(f"Session配置:")
    print(f"  - SECURE: {app.config.get('SESSION_COOKIE_SECURE')}")
    print(f"  - HTTPONLY: {app.config.get('SESSION_COOKIE_HTTPONLY')}")
    print(f"  - SAMESITE: {app.config.get('SESSION_COOKIE_SAMESITE')}")
    print(f"  - LIFETIME: {app.config.get('PERMANENT_SESSION_LIFETIME')}秒")
    
    try:
        print("\n启动服务器...")
        app.run(
            host=host,
            port=port,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"启动失败: {e}")
        sys.exit(1) 