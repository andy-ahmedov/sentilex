import threading
import time
import urllib.request

import webview
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


def wait_for_server(url, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"Flask server did not start within {timeout} seconds")


def main():
    server_thread = FlaskServerThread(flask_app, HOST, PORT)
    server_thread.start()
    wait_for_server(APP_URL)

    window = webview.create_window("Sentilex", APP_URL)
    window.events.closed += server_thread.shutdown

    try:
        webview.start()
    finally:
        server_thread.shutdown()
        server_thread.join(timeout=3)


if __name__ == "__main__":
    main()
