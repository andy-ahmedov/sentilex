#!/usr/bin/env python3
"""Smoke test: exported standalone PY reproduces the TXT output."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app as web_app  # noqa: E402


def main() -> int:
    input_path = PROJECT_ROOT / "texts" / "chast__7.txt"
    if not input_path.exists():
        print(f"[smoke-export] missing input file: {input_path}")
        return 1

    filename = secure_filename(input_path.name)
    artifacts = web_app.process_text(
        str(input_path),
        filename,
        50,
        export_script_file=True,
    )

    reference_txt = Path(artifacts["result_txt"])
    exported_script = Path(artifacts["result_py"])
    if not exported_script.exists():
        print(f"[smoke-export] missing exported script: {exported_script}")
        return 1

    script_text = exported_script.read_text(encoding="utf-8")
    if "SENTILEX_EXPORT_SCRIPT" not in script_text:
        print("[smoke-export] exported script marker not found")
        return 1
    if "SHOW_ONLY_EVEN_SECTION_LABELS" not in script_text:
        print("[smoke-export] plot customization block not found")
        return 1

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "standalone-results"
        completed = subprocess.run(
            [
                sys.executable,
                str(exported_script),
                str(input_path),
                "--chunk-size",
                "50",
                "--output-dir",
                str(output_dir),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr)
            print("[smoke-export] exported script execution failed")
            return 1

        generated_txt = output_dir / f"{input_path.name}_results.txt"
        generated_png = output_dir / f"{input_path.name}_sentiment_curve.png"
        if not generated_txt.exists() or not generated_png.exists():
            print("[smoke-export] exported script did not create TXT/PNG outputs")
            return 1
        if generated_png.stat().st_size == 0:
            print("[smoke-export] exported PNG is empty")
            return 1
        if reference_txt.read_text(encoding="utf-8") != generated_txt.read_text(encoding="utf-8"):
            print("[smoke-export] TXT output differs from main pipeline")
            return 1

    print("[smoke-export] OK: exported standalone script reproduces TXT output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
