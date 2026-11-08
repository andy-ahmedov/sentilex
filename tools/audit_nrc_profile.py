#!/usr/bin/env python3
"""Read-only, corpus-independent audit of the local NRC source and literary rules."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with contextlib.redirect_stdout(sys.stderr):
    import app
from scripts import emotions


def main():
    morph = app.functions.morph_vocab
    raw, _ = emotions.load_lexicon(morph, profile='raw')
    edited, _ = emotions.load_lexicon(morph, profile='literary')
    rows = list(csv.DictReader(io.StringIO(emotions.LEXICON_PATH.read_text(encoding='utf-8')), delimiter='\t'))
    _, changes = emotions.apply_literary_profile(rows)
    candidates = []
    for row in rows:
        categories = [key for key in emotions.KEYS if row[key] == '1']
        surface = emotions.TOKEN.findall(row['Russian Word'].lower().replace('ё', 'е'))
        if not categories or len(surface) != 1:
            continue
        parses = morph.parse(surface[0])
        if not {p.tag.POS for p in parses} & {'PREP', 'CONJ', 'PRCL', 'NPRO'}:
            continue
        candidates.append(dict(english=row['English Word'], russian=row['Russian Word'],
                               categories=categories,
                               readings=sorted({f'{p.normal_form}:{p.tag.POS}' for p in parses})))
    print(json.dumps(dict(source_rows=len(rows), source_sha256=emotions.SOURCE_SHA256,
                          raw_keys=len(raw), literary_keys=len(edited),
                          profile=emotions.profile_metadata('literary'),
                          candidate_count=len(candidates), candidates=candidates,
                          change_count=len(changes), changes=changes), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
