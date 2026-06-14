#!/usr/bin/env python3
"""Headless desktop layout smoke test for multiple screen sizes."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QFrame, QLabel, QPlainTextEdit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop.main import MainWindowController, load_window, resource_path  # noqa: E402


def validate_size(app: QApplication, width: int, height: int, expected_mode: str) -> None:
    window = load_window()
    window.setStyleSheet(resource_path("desktop", "style.qss").read_text(encoding="utf-8"))
    controller = MainWindowController(window)

    window.resize(width, height)
    window.show()
    for _ in range(3):
        app.processEvents()

    control_panel = window.findChild(QFrame, "controlPanel")
    results_panel = window.findChild(QFrame, "resultsPanel")
    plot_preview = window.findChild(QLabel, "plotPreview")
    text_preview = window.findChild(QPlainTextEdit, "resultTextPreview")
    if not all([control_panel, results_panel, plot_preview, text_preview]):
        raise RuntimeError("Missing expected desktop widgets for layout smoke")

    actual_mode = controller.current_layout_mode
    if actual_mode != expected_mode:
        raise RuntimeError(
            f"Unexpected layout mode for {width}x{height}: {actual_mode!r}, expected {expected_mode!r}"
        )

    if expected_mode == "wide" and control_panel.geometry().right() >= results_panel.geometry().left():
        raise RuntimeError(f"Wide layout panels overlap horizontally at {width}x{height}")

    if expected_mode == "stacked" and control_panel.geometry().bottom() >= results_panel.geometry().top():
        raise RuntimeError(f"Stacked layout panels overlap vertically at {width}x{height}")

    if plot_preview.geometry().intersects(text_preview.geometry()):
        raise RuntimeError(f"Result preview widgets overlap at {width}x{height}")

    if plot_preview.height() < 140:
        raise RuntimeError(f"Plot preview height too small at {width}x{height}: {plot_preview.height()}")

    if text_preview.height() < 120:
        raise RuntimeError(f"Text preview height too small at {width}x{height}: {text_preview.height()}")

    window.close()
    app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    validate_size(app, 1024, 720, "stacked")
    validate_size(app, 1366, 768, "wide")
    validate_size(app, 1920, 1080, "wide")

    print("[smoke-layout] OK: desktop layout fits stacked/wide modes without overlap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
