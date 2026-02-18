import threading
import time
import socket
import webbrowser
import platform
import subprocess
import os
from pathlib import Path

import webview
from webview.errors import WebViewException
from werkzeug.serving import make_server

from app import app as flask_app

HOST = "127.0.0.1"
PORT = 5000
APP_URL = f"http://{HOST}:{PORT}"


class FlaskServerThread(threading.Thread):
    def __init__(self, app, host, port):
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)
        self._context = app.app_context()
        self._context.push()
        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

    def run(self):
        self._server.serve_forever()

    def shutdown(self):
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            self._server.shutdown()
            self._context.pop()


def wait_for_server(host, port, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Flask server did not start on {host}:{port} within {timeout} seconds")


def print_linux_backend_help():
    print(
        "pywebview backend is not available.\n"
        "For Linux (Debian/Ubuntu) install system deps and retry:\n"
        "  sudo apt update\n"
        "  sudo apt install -y libnss3 libnspr4 libasound2 libxkbfile1 "
        "libxkbcommon-x11-0 libxcb-xinerama0 libxcb-cursor0 "
        "libxcb-icccm4 libxcb-keysyms1 libxcb-randr0 "
        "libxcb-render-util0 libxcb-shape0 libxcb-xfixes0 libegl1 libopengl0\n"
        "Optional GTK backend:\n"
        "  sudo apt install -y python3-gi gir1.2-webkit2-4.1\n"
    )


def fallback_to_browser():
    print_linux_backend_help()
    print(f"Fallback: opening browser at {APP_URL}")
    webbrowser.open(APP_URL)
    print("Browser fallback mode is running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def linux_qt_backend_ready():
    if platform.system() != "Linux":
        return True

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False

    candidate_plugins = [
        Path(".venv/lib/python3.12/site-packages/PyQt6/Qt6/plugins/platforms/libqxcb.so"),
        Path(".venv/lib/python3.12/site-packages/PyQt5/Qt5/plugins/platforms/libqxcb.so"),
    ]

    for plugin_path in candidate_plugins:
        if not plugin_path.exists():
            continue
        probe = subprocess.run(
            ["ldd", str(plugin_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0 and "not found" not in probe.stdout:
            return True

    return False


def main():
    server_thread = FlaskServerThread(flask_app, HOST, PORT)
    server_thread.start()
    wait_for_server(HOST, PORT)

    try:
        if platform.system() == "Linux" and not linux_qt_backend_ready():
            fallback_to_browser()
            return

        window = webview.create_window(
            "Sentilex",
            APP_URL,
            min_size=(1024, 700),
            background_color="#070d2c",
        )
        window.events.closed += server_thread.shutdown
        webview.start(gui="qt" if platform.system() == "Linux" else None)
    except WebViewException:
        fallback_to_browser()
    finally:
        server_thread.shutdown()
        server_thread.join(timeout=3)


if __name__ == "__main__":
    main()
