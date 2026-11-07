#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Use python3 if available, else python (compileall is optional but useful)
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYTHON_BIN="python3";
  elif command -v python >/dev/null 2>&1; then PYTHON_BIN="python";
  else PYTHON_BIN=""; fi
fi

echo "[verify] start"

if [[ -n "$PYTHON_BIN" ]]; then
  echo "[verify] python compileall"
  "$PYTHON_BIN" -m compileall "$ROOT/app.py" "$ROOT/scripts" >/dev/null
else
  echo "[verify] python not found -> skip compileall"
fi

echo "[verify] chunk TXT precision (Docker Python 3.10 + host Python when available)"
# Run the host check only when its analysis dependencies are installed.
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/tools/smoke_chunk_precision.py"
fi

echo "[verify] golden e2e regression"
"$ROOT/tools/golden_e2e.sh"

echo "[verify] OK"
