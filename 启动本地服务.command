#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "正在启动 Healthcare，首次加载模型需要一些时间。"
echo "浏览器地址：http://127.0.0.1:${PORT:-5050}"
echo "在此终端按 Control+C 可停止服务。"
exec .venv/bin/python start_local.py
