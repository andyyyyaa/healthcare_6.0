# 本地运行

环境：macOS Apple Silicon，Python 3.12，项目内 `.venv`。

## 启动

双击 `启动本地服务.command`，等待模型加载，然后访问：

http://127.0.0.1:5050

也可以在终端运行：

```bash
cd ~/Desktop/healthcare_6.0
.venv/bin/python start_local.py
```

在运行服务的终端按 Control+C，或双击 `停止本地服务.command` 停止。
如使用 README 中的后台启动命令，日志位于 `.local/server.log`。
默认仅监听本机；5000 端口已被 macOS 使用，因此选择 5050。
如需改端口：`PORT=5051 .venv/bin/python start_local.py`。

## 重新安装环境

```bash
cd ~/Desktop/healthcare_6.0
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 保存了本次验证通过的完整依赖版本。
心脏预测模型由 scikit-learn 1.6.1 保存，因此使用同一版本加载。
骨折模型使用 CPU 推理。
首次启动会下载官方基础权重，并缓存到 `.local/torch`。
从 GitHub 克隆的新环境还需按照 [README](README.md) 下载模型，
然后运行 `.venv/bin/python init_database.py` 创建空数据库。

本地会话密钥自动保存在 `.local/secret_key`，请勿分享该文件。
现有 `user_db.sqlite3` 和模型文件继续使用。
不要运行 `create_database.py`：它会删除并重建现有数据表。
