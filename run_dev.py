from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    frontend_dir = root / "frontend"

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "silo_blend.api.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
    ]

    frontend_cmd = ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"]

    print("Starting fullstack dev environment...")
    print("Backend:  http://127.0.0.1:8000")
    print("Frontend: http://127.0.0.1:5173")

    backend_proc = subprocess.Popen(backend_cmd, cwd=str(root), env=os.environ.copy())
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(frontend_dir), env=os.environ.copy())

    try:
        backend_code = backend_proc.wait()
        if frontend_proc.poll() is None:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=10)
        return backend_code
    except KeyboardInterrupt:
        print("\nStopping services...")
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
        for proc in (backend_proc, frontend_proc):
            if proc.poll() is None:
                proc.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
