#!/usr/bin/env python3
"""Render PySide6 UI design variants into PNG previews."""

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
DESIGN_DIR = ROOT / "desktop" / "design_variants"
MAIN_UI = ROOT / "desktop" / "ui" / "main.ui"
MAIN_QSS = ROOT / "desktop" / "style.qss"
OUTPUT_DIR = ROOT / "docs" / "design"

VARIANTS = {
    "A": "variant_A.ui",
    "B": "variant_B.ui",
    "C": "variant_C.ui",
}


def load_ui(ui_path: Path):
    loader = QUiLoader()
    ui_file = QFile(str(ui_path))
    if not ui_file.open(QFile.ReadOnly):
        raise RuntimeError(f"Cannot open UI file: {ui_path}")
    widget = loader.load(ui_file)
    ui_file.close()
    if widget is None:
        raise RuntimeError(f"Cannot load UI file: {ui_path}; loader error: {loader.errorString()}")
    return widget


def to_main_window(widget) -> QMainWindow:
    if isinstance(widget, QMainWindow):
        return widget
    window = QMainWindow()
    window.setCentralWidget(widget)
    return window


def render_variant(app: QApplication, variant_name: str, ui_path: Path, output_path: Path) -> None:
    widget = load_ui(ui_path)
    window = to_main_window(widget)
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    app.processEvents()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = window.grab()
    if pixmap.isNull():
        window.close()
        raise RuntimeError(f"Failed to capture window for variant {variant_name}")

    if not pixmap.save(str(output_path)):
        window.close()
        raise RuntimeError(f"Failed to save PNG: {output_path}")

    window.close()
    app.processEvents()


def render_parity_main(app: QApplication, ui_path: Path, qss_path: Path, output_path: Path) -> None:
    widget = load_ui(ui_path)
    window = to_main_window(widget)
    window.resize(1280, 800)

    if qss_path.exists():
        window.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError(f"Cannot open style file: {qss_path}")

    window.show()
    app.processEvents()
    app.processEvents()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = window.grab()
    if pixmap.isNull():
        window.close()
        raise RuntimeError("Failed to capture parity window")

    if not pixmap.save(str(output_path)):
        window.close()
        raise RuntimeError(f"Failed to save PNG: {output_path}")

    window.close()
    app.processEvents()


def main() -> int:
    missing = [DESIGN_DIR / filename for filename in VARIANTS.values() if not (DESIGN_DIR / filename).exists()]
    if missing:
        print("Missing UI files:")
        for path in missing:
            print(f"  - {path}")
        return 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    print(f"Rendering variants from {DESIGN_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"QT_QPA_PLATFORM={os.environ.get('QT_QPA_PLATFORM', '(default)')}")

    for variant_name, filename in VARIANTS.items():
        ui_path = DESIGN_DIR / filename
        output_path = OUTPUT_DIR / f"variant_{variant_name}.png"
        render_variant(app, variant_name, ui_path, output_path)
        print(f"[OK] {ui_path} -> {output_path}")

    if not MAIN_UI.exists():
        print(f"Missing parity UI file: {MAIN_UI}")
        return 1

    parity_output = OUTPUT_DIR / "desktop_parity.png"
    render_parity_main(app, MAIN_UI, MAIN_QSS, parity_output)
    print(f"[OK] {MAIN_UI} + {MAIN_QSS} -> {parity_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
