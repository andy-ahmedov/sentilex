#!/usr/bin/env python3
"""Preserve historical TXT rounding across Python versions and exports."""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import functions
from export_script import EXPORT_SCRIPT_TEMPLATE


def main():
    scores = [0, 0.3, 0.1, 0.5, 0, 0.05, 0, 1, 0, 0.2]
    exported = {"__name__": "export_precision_check"}
    template = EXPORT_SCRIPT_TEMPLATE.replace("__DEFAULT_CHUNK_SIZE__", "10")
    exec(compile(template, "export_template", "exec"), exported)
    for namespace in (vars(functions), exported):
        with patch.dict(namespace, calculate_sentiment=lambda sentence, *_: scores[int(sentence)]):
            values = namespace["calculate_sentences_sentiments"](
                [str(i) for i in range(len(scores))], [], 10, {}, {}
            )
        assert len(values) == 10
        assert all(f"{value:+.2f}" == "+0.22" for value in values), values
    print("[smoke-precision] OK: runtime/export preserve historical TXT rounding")


if __name__ == "__main__":
    main()
