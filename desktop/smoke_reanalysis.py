#!/usr/bin/env python3
"""Non-GUI smoke test: run analysis twice and verify outputs are refreshed."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as web_app  # noqa: E402


def run_once(input_path: Path, chunk_size: int) -> tuple[Path, Path, Path, int, int, int]:
    filename = secure_filename(input_path.name)
    artifacts = web_app.process_text(
        str(input_path),
        filename,
        chunk_size,
        export_script_file=True,
    )

    result_txt = Path(artifacts["result_txt"])
    result_png = Path(artifacts["result_png"])
    result_py = Path(artifacts["result_py"])

    if not result_txt.exists() or not result_png.exists() or not result_py.exists():
        raise RuntimeError("Result files were not generated")
    if result_txt.stat().st_size == 0:
        raise RuntimeError("Result TXT is empty")
    if result_png.stat().st_size == 0:
        raise RuntimeError("Result PNG is empty")
    if result_py.stat().st_size == 0:
        raise RuntimeError("Result PY is empty")
    txt_lines = result_txt.read_text(encoding="utf-8").splitlines()
    sentiment_lines = [line for line in txt_lines if line.strip()]
    if not sentiment_lines:
        raise RuntimeError("Result TXT has no content lines")
    if not any(line.startswith("+") or line.startswith("-") for line in sentiment_lines):
        raise RuntimeError("Result TXT format check failed: no signed sentiment lines found")
    script_text = result_py.read_text(encoding="utf-8")
    if "SENTILEX_EXPORT_SCRIPT" not in script_text:
        raise RuntimeError("Exported PY marker not found")

    return (
        result_txt,
        result_png,
        result_py,
        result_txt.stat().st_mtime_ns,
        result_png.stat().st_mtime_ns,
        result_py.stat().st_mtime_ns,
    )


def main() -> int:
    os.chdir(PROJECT_ROOT)
    input_path = PROJECT_ROOT / "texts" / "chast__7.txt"
    if not input_path.exists():
        print(f"[smoke] missing input file: {input_path}")
        return 1

    txt_path, png_path, py_path, txt_mtime_1, png_mtime_1, py_mtime_1 = run_once(
        input_path,
        chunk_size=50,
    )
    time.sleep(1.1)
    _, _, _, txt_mtime_2, png_mtime_2, py_mtime_2 = run_once(input_path, chunk_size=60)

    if txt_mtime_2 <= txt_mtime_1:
        print(f"[smoke] TXT did not refresh: {txt_path}")
        return 1
    if png_mtime_2 <= png_mtime_1:
        print(f"[smoke] PNG did not refresh: {png_path}")
        return 1
    if py_mtime_2 <= py_mtime_1:
        print(f"[smoke] PY did not refresh: {py_path}")
        return 1

    print("[smoke] OK: repeated analysis refreshed TXT, PNG and PY outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
