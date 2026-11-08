"""Bundled RusEmoLex, counting, density units, reports and step-curve checks."""
import csv
import hashlib
import sys
import tempfile
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
    data = (emotions.DATA_DIR / 'RusEmoLex.csv').read_bytes()
    assert hashlib.sha256(data).hexdigest() == emotions.SOURCE_SHA256
    rows = list(csv.DictReader(data.decode('utf-8').splitlines(), delimiter=';'))
    assert len(rows) == 1024 and {r['Класс'] for r in rows} == set(emotions.LABELS)
    loaded, digest = emotions.load_lexicon(morph)
    # All source rows contribute; inflectional normalization coalesces 25 entries.
    expected = {}
    for row in rows:
        key = tuple(emotions.tokens(row['Слово'], morph, {}))
        expected.setdefault(key, set()).add(emotions.LABELS.index(row['Класс']))
    assert loaded == expected and len(loaded) == 999
    assert set.union(*loaded.values()) == set(range(5))
    examples = emotions.count_chunks(['Злость грусть радость страх удивление.',
                                      'Я испугался. Без ума, без ума!'], 1, loaded, morph)
    assert examples[0]['counts'] == [1] * 5  # source occurrence metadata is not a weight
    assert examples[0]['densities'] == [.2] * 5
    assert examples[0]['leaders'] == list(range(5))
    assert examples[1]['counts'] == [0, 0, 2, 1, 0]
    assert [m['text'] for m in examples[1]['sentences'][0]['markers']] == ['испугался', 'Без ума', 'без ума']
    merged = emotions.normalize_entries([
        {'Слово': 'страх', 'Класс': 'Страх'}, {'Слово': 'страхи', 'Класс': 'Страх'},
        {'Слово': 'страх', 'Класс': 'Грусть'}], morph)
    assert merged == {('страх',): {1, 3}}
    lexicon = {('страх',): {3}, ('радость',): {2}, ('ужас',): {0, 3}, ('сильный', 'страх'): {3}}
    sentences = ['Страх страх.', 'Радость стол окно книга.', 'Сильный страх ужас.', 'Стол.', '!!!']
    chunks = emotions.count_chunks(sentences, 2, lexicon, morph)
    assert chunks[0]['counts'] == [0, 0, 1, 2, 0] and chunks[0]['words'] == 6
    assert chunks[0]['densities'] == [0, 0, 1 / 6, 1 / 3, 0]
    assert chunks[0]['sentences'][0]['densities'][3] == 1
    assert chunks[0]['sentences'][1]['densities'][2] == .25
    assert chunks[0]['leaders'] == [3] and chunks[0]['sentences'][1]['leaders'] == [2]
    phrase = chunks[1]['sentences'][0]
    assert [m['text'] for m in phrase['markers']] == ['Сильный страх', 'ужас']
    assert phrase['counts'] == [1, 0, 0, 2, 0] and phrase['words'] == 3
    assert chunks[-1]['words'] == 0 and chunks[-1]['densities'] == [0] * 5
    assert not chunks[-1]['leaders']
    unequal = emotions.count_chunks(['Страх.', 'Страх и стол у окна.'], 1, lexicon, morph)
    assert unequal[0]['densities'][3] == 1 and unequal[1]['densities'][3] == .2
    assert unequal[0]['leaders'] == unequal[1]['leaders'] == [3]
    for size in (1, 2, 3, 5):
        for chunk in emotions.count_chunks(sentences, size, lexicon, morph):
            assert chunk['words'] == sum(s['words'] for s in chunk['sentences'])
            assert chunk['counts'] == [sum(s['counts'][i] for s in chunk['sentences']) for i in range(5)]
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
        assert '[страх]' in txt and 'Маркеров на 100 слов' in txt
        html = Path(result['html']).read_text(encoding='utf-8')
        assert emotions.SOURCE_URL in html and '<details>' in html
        with open(result['csv'], encoding='utf-8-sig', newline='') as f:
            table = list(csv.reader(f, delimiter=';'))
        assert len(table) == 4 and len(table[0]) == 16
        assert table[1][4] == '6' and float(table[1][13]) == 1 / 3
        with open(result['sentences_csv'], encoding='utf-8-sig', newline='') as f:
            table = list(csv.reader(f, delimiter=';'))
        assert len(table) == len(sentences) + 1 and table[-1][2] == '!!!'
        assert table[-1][3] == '0' and not table[-1][-1]
        assert all(float(v) == 0 for v in table[-1][9:14])
    fig, ax = plt.subplots()
    emotions.draw_density(ax, chunks)
    assert len(ax.patches) == 5 and len(ax.lines) == len(chunks) + 1
    for i, patch in enumerate(ax.patches):
        values, edges, baseline = patch.get_data()
        assert np.allclose(values, [100 * c['densities'][i] for c in chunks])
        assert list(edges) == [.5, 2.5, 4.5, 5.5] and baseline is None
    assert ax.get_ylabel() == 'Маркеров на 100 слов'
    assert all(float(tick).is_integer() for tick in ax.get_xticks())
    plt.close(fig)
    print('[smoke-emotions] OK: 1024 rows / 999 normalized entries, five classes, densities and six artifacts')


if __name__ == '__main__':
    main()
