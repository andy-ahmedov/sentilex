#!/usr/bin/env python3
"""Render desktop main.ui parity preview into docs/design/desktop_parity.png."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QMainWindow

ROOT = Path(__file__).resolve().parents[1]
MAIN_UI = ROOT / "desktop" / "ui" / "main.ui"
MAIN_QSS = ROOT / "desktop" / "style.qss"
OUTPUT = ROOT / "docs" / "design" / "desktop_parity.png"


def load_main_window(path: Path) -> QMainWindow:
    loader = QUiLoader()
    ui_file = QFile(str(path))
    if not ui_file.open(QFile.ReadOnly):
        raise RuntimeError(f"Cannot open UI file: {path}")
    widget = loader.load(ui_file)
    ui_file.close()
    if widget is None:
        raise RuntimeError(f"Cannot load UI file: {path}; loader error: {loader.errorString()}")
    if isinstance(widget, QMainWindow):
        return widget
    window = QMainWindow()
    window.setCentralWidget(widget)
    return window


def main() -> int:
    if not MAIN_UI.exists():
        raise FileNotFoundError(f"Missing UI file: {MAIN_UI}")
    if not MAIN_QSS.exists():
        raise FileNotFoundError(f"Missing QSS file: {MAIN_QSS}")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = load_main_window(MAIN_UI)
    window.resize(1280, 820)
    window.setStyleSheet(MAIN_QSS.read_text(encoding="utf-8"))
    window.show()
    app.processEvents()
    app.processEvents()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pixmap = window.grab()
    if pixmap.isNull():
        raise RuntimeError("Failed to capture main window")
    if not pixmap.save(str(OUTPUT)):
        raise RuntimeError(f"Failed to write {OUTPUT}")

    print(f"[OK] {MAIN_UI} + {MAIN_QSS} -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
