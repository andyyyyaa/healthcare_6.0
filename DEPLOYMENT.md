# 生产环境部署说明

## 问题诊断

### Flash失效和登录状态问题的原因
1. **Session配置问题**: 生产环境下session配置不正确
2. **Secret Key问题**: 环境变量未正确设置
3. **Cookie配置问题**: HTTPS/HTTP配置不匹配

## 解决方案

### 1. 使用新的启动脚本
```bash
# 使用生产环境启动脚本
python3 start_server.py
```

### 2. 设置环境变量
```bash
# 设置Secret Key (重要!)
export SECRET_KEY="your-very-secure-secret-key-here"

# 设置主机和端口
export HOST="0.0.0.0"
export PORT="5000"
```

### 3. 检查数据库权限
确保服务器上的数据库文件有正确的读写权限：
```bash
chmod 644 user_db.sqlite3
chmod 755 static/
```

### 4. 防火墙配置
确保服务器防火墙允许5000端口：
```bash
# Ubuntu/Debian
sudo ufw allow 5000

# CentOS/RHEL
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

## 部署步骤

### 1. 上传代码到服务器
```bash
# 上传所有文件到服务器
scp -r ./* user@your-server:/path/to/app/
```

### 2. 安装依赖
```bash
# 在服务器上安装依赖
pip3 install -r requirements.txt
```

### 3. 设置环境变量
```bash
# 编辑 ~/.bashrc 或 ~/.profile
echo 'export SECRET_KEY="your-secret-key"' >> ~/.bashrc
echo 'export HOST="0.0.0.0"' >> ~/.bashrc
echo 'export PORT="5000"' >> ~/.bashrc
source ~/.bashrc
```

### 4. 启动应用
```bash
# 使用生产环境启动脚本
python3 start_server.py
```

### 5. 后台运行 (推荐)
```bash
# 使用nohup后台运行
nohup python3 start_server.py > app.log 2>&1 &

# 或者使用screen
screen -S healthcare
python3 start_server.py
# Ctrl+A, D 分离screen
```

## 故障排除

### 1. 检查日志
```bash
# 查看应用日志
tail -f app.log

# 查看系统日志
sudo journalctl -u your-service -f
```

### 2. 检查进程
```bash
# 查看Python进程
ps aux | grep python

# 查看端口占用
netstat -tlnp | grep 5000
```

### 3. 测试连接
```bash
# 本地测试
curl http://localhost:5000

# 远程测试
curl http://your-server-ip:5000
```

## 安全建议

1. **使用HTTPS**: 生产环境建议使用HTTPS
2. **设置强Secret Key**: 使用随机生成的强密钥
3. **限制访问**: 使用防火墙限制访问来源
4. **定期备份**: 定期备份数据库和代码
5. **监控日志**: 监控应用日志和系统日志

## 性能优化

1. **使用Gunicorn**: 生产环境推荐使用Gunicorn
2. **Nginx反向代理**: 使用Nginx作为前端代理
3. **数据库优化**: 考虑使用PostgreSQL等生产级数据库
4. **缓存**: 添加Redis缓存支持 