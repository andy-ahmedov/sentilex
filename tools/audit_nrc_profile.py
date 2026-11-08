#!/usr/bin/env python3
"""Read-only, corpus-independent audit of the local NRC source and literary rules."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
with contextlib.redirect_stdout(sys.stderr):
    import app
from scripts import emotions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path,
                        help='Optional pinned official NRC ZIP for sense evidence')
    args = parser.parse_args()
    morph = app.functions.morph_vocab
    raw, _ = emotions.load_lexicon(morph, profile='raw')
    edited, _ = emotions.load_lexicon(morph, profile='literary')
    first, _ = emotions.load_lexicon(morph, profile='literary_v1')
    rows = list(csv.DictReader(io.StringIO(emotions.LEXICON_PATH.read_text(encoding='utf-8')), delimiter='\t'))
    _, changes = emotions.apply_literary_profile(rows)
    candidates, comma_lists, groups, cache = [], [], {}, {}
    for row in rows:
        categories = [key for key in emotions.KEYS if row[key] == '1']
        if categories:
            key = tuple(emotions.tokens(row['Russian Word'], morph, cache))
            groups.setdefault(key, []).append(dict(english=row['English Word'],
                                                   russian=row['Russian Word'],
                                                   categories=categories))
            if ',' in row['Russian Word']:
                comma_lists.append(dict(english=row['English Word'], russian=row['Russian Word']))
        surface = emotions.TOKEN.findall(row['Russian Word'].lower().replace('ё', 'е'))
        if not categories or len(surface) != 1:
            continue
        parses = morph.parse(surface[0])
        if not {p.tag.POS for p in parses} & {'PREP', 'CONJ', 'PRCL', 'NPRO'}:
            continue
        candidates.append(dict(english=row['English Word'], russian=row['Russian Word'],
                               categories=categories,
                               readings=sorted({f'{p.normal_form}:{p.tag.POS}' for p in parses})))
    collisions = [{'normalized': ' '.join(key), 'entries': entries}
                  for key, entries in groups.items() if len(entries) > 1]
    result = dict(source_rows=len(rows), source_sha256=emotions.SOURCE_SHA256,
                  raw_keys=len(raw), literary_keys=len(edited),
                  literary_v1_keys=len(first),
                  profile=emotions.profile_metadata('literary'),
                  v1_profile=emotions.profile_metadata('literary_v1'),
                  candidate_count=len(candidates), candidates=candidates,
                  comma_lists=comma_lists, collision_count=len(collisions),
                  collisions=collisions, change_count=len(changes), changes=changes)
    if args.archive:
        evidence = emotions.read_literary_profile()['sense_evidence']
        if hashlib.sha256(args.archive.read_bytes()).hexdigest() != evidence['archive_sha256']:
            raise ValueError('NRC archive checksum mismatch')
        with zipfile.ZipFile(args.archive) as archive:
            data = archive.read(evidence['member'])
        if hashlib.sha256(data).hexdigest() != evidence['sha256']:
            raise ValueError('NRC sense-level checksum mismatch')
        selected = {c['english'] for c in changes} | {'blue', 'labor', 'honest', 'intelligence'}
        senses = {}
        for sense, category, value in csv.reader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t'):
            if sense.split('--')[0] in selected and value == '1':
                senses.setdefault(sense, []).append(category)
        result['sense_evidence'] = dict(evidence, checked=True, selected_senses=senses)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
