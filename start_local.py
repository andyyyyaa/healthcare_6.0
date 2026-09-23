"""Run the existing application locally with its project virtual environment."""

import os
from pathlib import Path
import secrets
import atexit
import fcntl

import certifi


def main():
    project = Path(__file__).resolve().parent
    os.chdir(project)
    runtime = project / ".local"
    runtime.mkdir(exist_ok=True, mode=0o700)
    # Keep the descriptor open for the lifetime of the server.
    server_lock = (runtime / "server.lock").open("a")
    try:
        fcntl.flock(server_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Healthcare 已在运行，请访问 http://127.0.0.1:5050")
        return
    pid_file = runtime / "server.pid"
    pid_file.write_text(str(os.getpid()) + "\n")
    atexit.register(lambda: pid_file.unlink(missing_ok=True))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(runtime / "matplotlib"))
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("TORCH_HOME", str(runtime / "torch"))
    if not os.environ.get("SECRET_KEY"):
        key_file = runtime / "secret_key"
        if not key_file.exists():
            with key_file.open("x", encoding="utf-8") as stream:
                key_file.chmod(0o600)
                stream.write(secrets.token_hex(32))
        os.environ["SECRET_KEY"] = key_file.read_text(encoding="utf-8").strip()

    from app import app

    port = int(os.environ.get("PORT", "5050"))
    print(f"Healthcare: http://127.0.0.1:{port}", flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
