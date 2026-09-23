#!/bin/bash
set -e
cd "$(dirname "$0")"
.venv/bin/python - <<'PY'
import os
from pathlib import Path
import shlex
import signal
import subprocess

pid_file = Path('.local/server.pid')
if not pid_file.exists():
    print('没有正在运行的本地服务。')
else:
    pid = int(pid_file.read_text().strip())
    command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout.strip()
    args = shlex.split(command)
    expected = str(Path('start_local.py').resolve())
    if expected in args or 'start_local.py' in args:
        os.kill(pid, signal.SIGTERM)
        print('已停止 Healthcare 本地服务。')
    else:
        print('记录的服务进程已退出。')
    pid_file.unlink(missing_ok=True)
PY
