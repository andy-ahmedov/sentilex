"""Exercise the golden upload/download over HTTP inside the Docker container."""

import difflib
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    input_path, baseline_path, chunk_size = sys.argv[1:]
    base_url = "http://127.0.0.1:5000"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(120):
        try:
            with opener.open(base_url, timeout=2) as response:
                assert response.status == 200
            break
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.25)
    else:
        raise RuntimeError("Flask did not become ready inside the container")

    boundary = "sentilex-golden-upload-boundary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"chunk_size\"\r\n\r\n"
        f"{chunk_size}\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="golden_input.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode() + Path(input_path).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        base_url + "/", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with opener.open(request, timeout=120) as response:
        html = response.read().decode("utf-8")
    links = re.findall(r'''href=["'](/download/[^"']+\.txt)["']''', html)
    if not links:
        raise RuntimeError("Upload response contains no TXT download link")
    with opener.open(urllib.parse.urljoin(base_url, links[0]), timeout=30) as response:
        actual = response.read()
    expected = Path(baseline_path).read_bytes()
    if actual != expected:
        diff = difflib.unified_diff(
            expected.decode("utf-8").splitlines(True),
            actual.decode("utf-8").splitlines(True),
            fromfile="baseline", tofile="downloaded TXT",
        )
        sys.stdout.writelines(diff)
        raise SystemExit(1)
    print("[golden] OK: HTTP-uploaded/downloaded TXT matches baseline (container network)")


if __name__ == "__main__":
    main()
