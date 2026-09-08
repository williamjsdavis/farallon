"""Start both local services, and stop our child processes together with Ctrl+C."""
from pathlib import Path
import os
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    for port in (8000, 3000):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                print(f"Port {port} is already in use. If Farallon is running, open http://127.0.0.1:3000.")
                return 1
    children = []
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8000"], cwd=ROOT, start_new_session=True))
        children.append(subprocess.Popen(["npm", "run", "dev"], cwd=ROOT / "web", start_new_session=True))
        ready = False
        deadline = time.monotonic() + 60
        while all(child.poll() is None for child in children):
            if not ready:
                if time.monotonic() > deadline:
                    print("Startup did not become ready within 60 seconds. See the service logs above.", flush=True)
                    return 1
                try:
                    with urlopen("http://127.0.0.1:3000/api/health", timeout=1) as response:
                        ready = response.status == 200
                    if ready:
                        print("\nFarallon is ready: http://127.0.0.1:3000\nPress Ctrl+C to stop both servers.\n", flush=True)
                except (OSError, TimeoutError):
                    pass
            time.sleep(0.4)
        print("A demo service exited. See the service logs above.", flush=True)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
