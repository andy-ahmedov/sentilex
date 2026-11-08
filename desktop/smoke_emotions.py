"""Bundled NRC 2022, counting, density units, reports and step-curve checks."""
import csv
import hashlib
import sys
import tempfile
from unittest.mock import patch as mock_patch
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import emotions
import app


def main():
    morph = app.functions.morph_vocab
    data = emotions.LEXICON_PATH.read_bytes()
    assert hashlib.sha256(data).hexdigest() == emotions.SOURCE_SHA256
    rows = list(csv.DictReader(data.decode('utf-8').splitlines(), delimiter='\t'))
    assert len(rows) == 14154 and all(key in rows[0] for key in emotions.KEYS)
    loaded, digest = emotions.load_lexicon(morph)
    expected = {}
    for row in rows:
        categories = {i for i, key in enumerate(emotions.KEYS) if row[key] == '1'}
        import re
        if categories and re.search('[а-яё]', row['Russian Word'].lower()):
            key = tuple(emotions.tokens(row['Russian Word'], morph, {}))
            if key:
                expected.setdefault(key, set()).update(categories)
    assert loaded == expected and len(loaded) == 3707
    assert set.union(*loaded.values()) == set(range(8))
    # Actual translations have multiple labels; union does not multiply occurrences.
    examples = emotions.count_chunks(['Я боюсь.', 'Стол и окно.', 'Страх страх.'], 1, loaded, morph)
    assert examples[0]['counts'] == [0, 0, 0, 1, 0, 0, 0, 0]
    assert examples[0]['densities'][3] == .5
    assert examples[1]['counts'] == [0] * 8 and not examples[1]['leaders']
    assert examples[2]['counts'] == [2, 0, 0, 2, 0, 2, 0, 0]
    assert examples[2]['leaders'] == [0, 3, 5]
    assert [m['text'] for m in examples[2]['sentences'][0]['markers']] == ['Страх', 'страх']
    def row(word, *keys):
        return {'Russian Word': word, **{k: str(int(k in keys)) for k in emotions.KEYS}}
    merged = emotions.normalize_entries([
        row('страх', 'fear'), row('страхи', 'fear'), row('страх', 'sadness'),
        row('стол'), row('only-positive'), row('table', 'joy')], morph)
    assert merged == {('страх',): {1, 3}}
    # The pinned loader must reject missing or substituted resources.
    with tempfile.TemporaryDirectory() as folder:
        damaged = Path(folder) / 'Russian-NRC-EmoLex.txt'
        with mock_patch.object(emotions, 'LEXICON_PATH', damaged):
            try:
                emotions.load_lexicon(morph)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing dictionary accepted')
            damaged.write_bytes(data[:-20])
            try:
                emotions.load_lexicon(morph)
            except ValueError:
                pass
            else:
                raise AssertionError('Tampered dictionary accepted')
    def vector(first_five):
        return first_five + [0, 0, 0]
    lexicon = {('страх',): {3}, ('радость',): {2}, ('ужас',): {0, 3}, ('сильный', 'страх'): {3}}
    sentences = ['Страх страх.', 'Радость стол окно книга.', 'Сильный страх ужас.', 'Стол.', '!!!']
    chunks = emotions.count_chunks(sentences, 2, lexicon, morph)
    assert chunks[0]['counts'] == vector([0, 0, 1, 2, 0]) and chunks[0]['words'] == 6
    assert chunks[0]['densities'] == vector([0, 0, 1 / 6, 1 / 3, 0])
    assert chunks[0]['sentences'][0]['densities'][3] == 1
    assert chunks[0]['sentences'][1]['densities'][2] == .25
    assert chunks[0]['leaders'] == [3] and chunks[0]['sentences'][1]['leaders'] == [2]
    phrase = chunks[1]['sentences'][0]
    assert [m['text'] for m in phrase['markers']] == ['Сильный страх', 'ужас']
    assert phrase['counts'] == vector([1, 0, 0, 2, 0]) and phrase['words'] == 3
    assert chunks[-1]['words'] == 0 and chunks[-1]['densities'] == [0] * 8
    assert not chunks[-1]['leaders']
    unequal = emotions.count_chunks(['Страх.', 'Страх и стол у окна.'], 1, lexicon, morph)
    assert unequal[0]['densities'][3] == 1 and unequal[1]['densities'][3] == .2
    assert unequal[0]['leaders'] == unequal[1]['leaders'] == [3]
    for size in (1, 2, 3, 5):
        for chunk in emotions.count_chunks(sentences, size, lexicon, morph):
            assert chunk['words'] == sum(s['words'] for s in chunk['sentences'])
            assert chunk['counts'] == [sum(s['counts'][i] for s in chunk['sentences']) for i in range(8)]
            assert chunk['leaders'] == emotions.find_leaders(chunk['counts'])
            for sentence in chunk['sentences']:
                for marker in sentence['markers']:
                    assert sentence['text'][marker['start']:marker['end']] == marker['text']
    assert emotions.find_leaders([.100000000001, .1, 0, 0, 0]) == [0]
    assert emotions.find_leaders([.1, 0, .1, 0, 0]) == [0, 2]
    escaped = emotions.count_chunks(['<script>alert("Страх")</script> & радость.'], 1, lexicon, morph)
    detail_txt, detail_html = emotions.sentence_report(escaped)
    assert '<script>' not in detail_html and '&lt;script&gt;' in detail_html and '&amp;' in detail_html
    assert 'Найдено: «Страх» → Страх; «радость» → Радость' in detail_txt
    assert 'В этом предложении не найдены слова из словаря' in emotions.sentence_report(chunks)[1]
    summary = emotions.summarize(chunks)
    assert abs(sum(summary['text_shares']) + summary['unknown_share'] - 1) < 1e-12
    assert abs(sum(summary['marker_shares']) - 1) < 1e-12
    assert emotions.summarize([])['unknown_share'] == 0
    assert len(summary['totals']) == len(summary['text_shares']) == len(summary['marker_shares']) == 8
    with tempfile.TemporaryDirectory() as folder:
        result = emotions.write_results(sentences, [.1, .1, 0, 0, -.1], 2,
                                        lexicon, digest, morph, Path(folder) / 'test')
        assert set(result) == {'png', 'txt', 'csv', 'html', 'density_png', 'sentences_csv'}
        assert all(Path(p).stat().st_size for p in result.values())
        for key in ('png', 'density_png'):
            assert Path(result[key]).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        txt = Path(result['txt']).read_text(encoding='utf-8')
        assert txt.startswith('Выделены слова из эмоционального словаря.')
        assert emotions.SOURCE_SHA256 in txt and emotions.PUBLICATION in txt
        assert 'Google Translate, август 2022' in txt and 'Ярошенко' not in txt
        assert '[страх]' in txt and 'Маркеров на 100 слов' in txt
        html = Path(result['html']).read_text(encoding='utf-8')
        assert emotions.SOURCE_URL in html and '<details>' in html
        with open(result['csv'], encoding='utf-8-sig', newline='') as f:
            table = list(csv.reader(f, delimiter=';'))
        assert len(table) == 4 and len(table[0]) == 22
        assert table[1][4] == '6' and float(table[1][16]) == 1 / 3
        with open(result['sentences_csv'], encoding='utf-8-sig', newline='') as f:
            table = list(csv.reader(f, delimiter=';'))
        assert len(table) == len(sentences) + 1 and table[-1][2] == '!!!'
        assert table[-1][3] == '0' and not table[-1][-1]
        assert all(float(v) == 0 for v in table[-1][12:20])
    fig, ax = plt.subplots()
    emotions.draw_density(ax, chunks)
    assert len(ax.patches) == 8 and len(ax.lines) == len(chunks) + 1
    for i, patch in enumerate(ax.patches):
        values, edges, baseline = patch.get_data()
        assert np.allclose(values, [100 * c['densities'][i] for c in chunks])
        assert list(edges) == [.5, 2.5, 4.5, 5.5] and baseline is None
    assert ax.get_ylabel() == 'Маркеров на 100 слов'
    assert all(float(tick).is_integer() for tick in ax.get_xticks())
    plt.close(fig)
    print('[smoke-emotions] OK: 14154 rows / 3707 normalized entries, eight classes, densities and six artifacts')


if __name__ == '__main__':
    main()
