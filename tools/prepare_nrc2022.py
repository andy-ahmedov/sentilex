#!/usr/bin/env python3
"""Prepare the pinned NRC 2022 resource for a local research build, never for Git."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def prepare(archive_path=None):
    metadata = json.loads((ROOT / 'scripts/data/NRC2022.SOURCE.json').read_text(encoding='utf-8'))
    if archive_path:
        archive = Path(archive_path).read_bytes()
    else:
        with urlopen(metadata['archive_url'], timeout=60) as response:
            archive = response.read()
    if hashlib.sha256(archive).hexdigest() != metadata['archive_sha256']:
        raise ValueError('Архив NRC отличается от закреплённой версии 2022 года')
    with ZipFile(io.BytesIO(archive)) as zipped:
        resources = {
            'Russian-NRC-EmoLex.txt': (zipped.read(metadata['archive_member']), metadata['sha256']),
            'README.txt': (zipped.read('NRC-Emotion-Lexicon/README.txt'), metadata['readme_sha256']),
        }
    for name, (data, expected) in resources.items():
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Контрольная сумма не совпала: {name}')
    destination = ROOT / 'scripts/data/nrc2022'
    destination.mkdir(parents=True, exist_ok=True)
    for name, (data, _) in resources.items():
        temporary = destination / (name + '.tmp')
        temporary.write_bytes(data)
        temporary.replace(destination / name)
    print(f'[prepare-nrc2022] OK: {metadata["source_rows"]} source rows, Google Translate August 2022; local research data in {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Use an already downloaded official ZIP')
    args = parser.parse_args()
    prepare(args.archive)
