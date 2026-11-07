#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- configurable knobs ---
IMAGE="${IMAGE:-sentilex_golden}"
CONTAINER="${CONTAINER:-sentilex_golden}"
HOST_PORT="${HOST_PORT:-5000}"
GOLDEN_HTTP_MODE="${GOLDEN_HTTP_MODE:-host}"
if [[ "$GOLDEN_HTTP_MODE" != host && "$GOLDEN_HTTP_MODE" != container ]]; then
  echo "[golden] ERROR: GOLDEN_HTTP_MODE must be host or container"
  exit 1
fi

# Your known golden scenario inputs
INPUT="${INPUT:-$ROOT/texts/chast__7.txt}"
BASELINE="${BASELINE:-$ROOT/results/chast_7_result.txt}"

# Fragment / chunk defaults
FRAGMENT_VALUE="${FRAGMENT_VALUE:-50}"
# If you know the exact field name, set it explicitly:
#   FRAGMENT_FIELD=chunk_size tools/golden_e2e.sh
FRAGMENT_FIELD="${FRAGMENT_FIELD:-}"

# If you know file field name, set explicitly; otherwise auto-detect from templates/upload.html
FILE_FIELD="${FILE_FIELD:-}"

# Debug: set DEBUG=1 to dump HTML to a temp file if something fails
DEBUG="${DEBUG:-0}"

need_file() {
  if [[ ! -f "$1" ]]; then
    echo "[golden] ERROR: missing file: $1"
    exit 1
  fi
}

need_file "$INPUT"
need_file "$BASELINE"

echo "[golden] build image: $IMAGE"
docker build -t "$IMAGE" "$ROOT" >/dev/null

echo "[golden] chunk TXT precision in container"
docker run --rm -v "$ROOT/tools/smoke_chunk_precision.py:/tmp/smoke_chunk_precision.py:ro" \
  -e PYTHONPATH=/app/scripts "$IMAGE" python /tmp/smoke_chunk_precision.py

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
echo "[golden] start container: $CONTAINER (port ${HOST_PORT}->5000)"
docker run -d --rm --name "$CONTAINER" -p "$HOST_PORT:5000" "$IMAGE" >/dev/null

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

if [[ "$GOLDEN_HTTP_MODE" == container ]]; then
  docker cp "$INPUT" "$CONTAINER:/tmp/golden_input.txt"
  docker cp "$BASELINE" "$CONTAINER:/tmp/golden_baseline.txt"
  docker exec -i "$CONTAINER" python - /tmp/golden_input.txt /tmp/golden_baseline.txt \
    "$FRAGMENT_VALUE" < "$ROOT/tools/golden_http.py"
  exit 0
fi

echo "[golden] wait for http://127.0.0.1:${HOST_PORT}/"
for _ in $(seq 1 120); do
  if curl --connect-timeout 2 --max-time 3 -fsS "http://127.0.0.1:${HOST_PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

TMPDIR="$(mktemp -d)"
COOKIEJAR="$TMPDIR/cookies.txt"
DOWNLOADED_TXT="$TMPDIR/downloaded_result.txt"
HTML_DUMP="$TMPDIR/response.html"

UPLOAD_HTML="$ROOT/templates/upload.html"

# ---- autodetect FILE_FIELD / FRAGMENT_FIELD from templates/upload.html (best-effort) ----
if [[ -f "$UPLOAD_HTML" ]]; then
  ONE_LINE="$(tr '\n' ' ' < "$UPLOAD_HTML")"

  if [[ -z "$FILE_FIELD" ]]; then
    FILE_TAG="$(printf '%s' "$ONE_LINE" | grep -oE '<input[^>]*type=["'\'']?file["'\'']?[^>]*>' | head -n1 || true)"
    if [[ -n "$FILE_TAG" ]]; then
      FILE_FIELD="$(printf '%s' "$FILE_TAG" | sed -nE 's/.*name=["'\'']([^"'\'']+)["'\''].*/\1/p')"
    fi
  fi

  if [[ -z "$FRAGMENT_FIELD" ]]; then
    FRAG_TAG="$(printf '%s' "$ONE_LINE" | grep -oE "<input[^>]*value=[\"']${FRAGMENT_VALUE}[\"'][^>]*>" | head -n1 || true)"
    if [[ -n "$FRAG_TAG" ]]; then
      FRAGMENT_FIELD="$(printf '%s' "$FRAG_TAG" | sed -nE 's/.*name=["'\'']([^"'\'']+)["'\''].*/\1/p')"
    fi
  fi
fi

FILE_FIELD="${FILE_FIELD:-file}"

echo "[golden] file field: ${FILE_FIELD}"
if [[ -n "$FRAGMENT_FIELD" ]]; then
  echo "[golden] fragment field: ${FRAGMENT_FIELD}=${FRAGMENT_VALUE}"
else
  echo "[golden] fragment field: (not detected) -> relying on backend default=${FRAGMENT_VALUE}"
fi

CURL_FORM=(-F "${FILE_FIELD}=@${INPUT}")
if [[ -n "$FRAGMENT_FIELD" ]]; then
  CURL_FORM+=(-F "${FRAGMENT_FIELD}=${FRAGMENT_VALUE}")
fi

echo "[golden] upload + analyze..."
HTML="$(curl -fsS -L -c "$COOKIEJAR" -b "$COOKIEJAR" "${CURL_FORM[@]}" "http://127.0.0.1:${HOST_PORT}/")"

if [[ "$DEBUG" == "1" ]]; then
  printf '%s' "$HTML" > "$HTML_DUMP"
  echo "[golden] debug: html saved to $HTML_DUMP"
fi

# Extract first /download/... link (prefer .txt)
DL_PATH="$(
  printf '%s' "$HTML" \
    | tr "'\"" '\n' \
    | grep -E '^/download/' \
    | grep -i '\.txt' \
    | head -n1 || true
)"

if [[ -z "$DL_PATH" ]]; then
  # fallback: any /download/
  DL_PATH="$(
    printf '%s' "$HTML" \
      | tr "'\"" '\n' \
      | grep -E '^/download/' \
      | head -n1 || true
  )"
fi

if [[ -z "$DL_PATH" ]]; then
  echo "[golden] ERROR: could not find /download/ link in response HTML."
  if [[ "$DEBUG" != "1" ]]; then
    printf '%s' "$HTML" > "$HTML_DUMP"
    echo "[golden] html saved to $HTML_DUMP"
  fi
  exit 2
fi

echo "[golden] download: $DL_PATH"
curl -fsS -L -b "$COOKIEJAR" "http://127.0.0.1:${HOST_PORT}${DL_PATH}" -o "$DOWNLOADED_TXT"

echo "[golden] diff vs baseline: $BASELINE"
diff -u "$BASELINE" "$DOWNLOADED_TXT" >/dev/null

echo "[golden] OK: downloaded TXT matches baseline"
