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
    print('[smoke-emotions] OK')


if __name__ == '__main__':
    main()
