#!/usr/bin/env python3
"""Headless desktop smoke test for chunk-size slider options."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop.main import MainWindowController, load_window, resource_path  # noqa: E402


def validate_case(
    app: QApplication,
    sentence_count: int,
    expected_sizes: list[int],
    expected_default: int,
) -> None:
    window = load_window()
    window.setStyleSheet(resource_path("desktop", "style.qss").read_text(encoding="utf-8"))
    controller = MainWindowController(window)
    window.show()
    app.processEvents()

    controller._configure_chunk_slider(sentence_count)
    app.processEvents()

    if controller.allowed_chunk_sizes != expected_sizes:
        raise RuntimeError(
            f"Unexpected chunk sizes for {sentence_count}: "
            f"{controller.allowed_chunk_sizes!r} != {expected_sizes!r}"
        )

    current_text = controller.chunk_current_label.text()
    expected_text = f"Текущее значение: {expected_default}"
    if current_text != expected_text:
        raise RuntimeError(
            f"Unexpected current label for {sentence_count}: {current_text!r} != {expected_text!r}"
        )

    max_text = controller.chunk_max_label.text()
    expected_max = str(expected_sizes[-1])
    if max_text != expected_max:
        raise RuntimeError(
            f"Unexpected max label for {sentence_count}: {max_text!r} != {expected_max!r}"
        )

    sentence_text = controller.sentence_count_label.text()
    if str(sentence_count) not in sentence_text:
        raise RuntimeError(
            f"Sentence count label does not mention {sentence_count}: {sentence_text!r}"
        )

    window.close()
    app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    validate_case(app, 7, [1, 2, 3, 4, 5, 6, 7], 7)
    validate_case(app, 50, [10, 20, 30, 40, 50], 50)
    validate_case(app, 237, list(range(10, 231, 10)) + [237], 50)

    print("[smoke-chunk] OK: chunk slider reflects sentence-count-based options")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
