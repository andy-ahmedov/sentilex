"""NRC counts, ties, shares and external archive regression."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import emotions
import app


def main():
    morph = app.functions.morph_vocab
    lexicon = {('страх',): {3}, ('радость',): {4}, ('ужас',): {0, 3}, ('сильный', 'страх'): {3}}
    chunks = emotions.count_chunks(['Страх радость.', 'Сильный страх ужас.', 'Стол.'], 1, lexicon, morph)
    assert chunks[0]['leaders'] == [3, 4]
    assert chunks[1]['counts'][3] == 2 and chunks[1]['counts'][0] == 1
    assert not chunks[2]['leaders']
    for chunk in chunks:
        assert chunk['counts'] == [sum(s['counts'][i] for s in chunk['sentences']) for i in range(8)]
        for sentence in chunk['sentences']:
            for marker in sentence['markers']:
                assert sentence['text'][marker['start']:marker['end']] == marker['text']
    phrase = chunks[1]['sentences'][0]
    assert [m['text'] for m in phrase['markers']] == ['Сильный страх', 'ужас']
    assert phrase['markers'][1]['categories'] == [0, 3]
    repeats = emotions.count_chunks(['Страх страх.', 'Радость радость радость.', 'Стол.'], 2, lexicon, morph)
    assert repeats[0]['leaders'] == [4]
    assert repeats[0]['sentences'][0]['leaders'] == [3]
    assert len(repeats[0]['sentences'][0]['markers']) == 2
    assert repeats[1]['start'] == repeats[1]['end'] == 3
    assert emotions.find_leaders([2, 0, 0, 2, 0, 2, 0, 0]) == [0, 3, 5]
    escaped = emotions.count_chunks(['<script>alert("Страх")</script> & радость.'], 1, lexicon, morph)
    detail_txt, detail_html = emotions.sentence_report(escaped)
    assert '<script>' not in detail_html and '&lt;script&gt;' in detail_html
    assert '&amp;' in detail_html and 'Лидеры предложения: Страх / Радость' in detail_txt
    assert 'Маркеры не найдены' in emotions.sentence_report(chunks)[1]
    summary = emotions.summarize(chunks)
    assert abs(sum(summary['text_shares']) + summary['unknown_share'] - 1) < 1e-12
    assert abs(sum(summary['marker_shares']) - 1) < 1e-12
    assert emotions.summarize([])['unknown_share'] == 0
    archive = Path('NRC-Suite-of-Sentiment-Emotion-Lexicons.zip')
    if archive.exists():
        loaded, digest = emotions.load_lexicon(archive, morph)
        assert len(loaded) > 1000 and set.union(*loaded.values()) == set(range(8))
        print('Imported normalized Russian entries:', len(loaded))
    with tempfile.TemporaryDirectory() as folder:
        result = emotions.write_results(['Страх радость.', 'Стол.'], [.1, 0], 1, lexicon, 'test', morph, Path(folder) / 'test')
        assert all(Path(p).stat().st_size for p in result.values())
        assert set(result) == {'png', 'txt', 'csv', 'html'}
        assert 'Маркер: «Страх»' in Path(result['txt']).read_text(encoding='utf-8')
        assert 'Предложение 2' in Path(result['html']).read_text(encoding='utf-8')
    print('[smoke-emotions] OK')


if __name__ == '__main__':
    main()
