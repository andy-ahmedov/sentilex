"""Independent checks for the conservative Russian literary NRC profile."""
import csv
import hashlib
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from scripts import emotions


def main():
    morph = app.functions.morph_vocab
    original = emotions.LEXICON_PATH.read_bytes()
    rows = list(csv.DictReader(original.decode().splitlines(), delimiter='\t'))
    before = deepcopy(rows)
    revised, changes = emotions.apply_literary_profile(rows)
    assert rows == before, 'Original rows were mutated'
    actions = {c['english']: c['action'] for c in changes}
    assert actions == {'wot': 'exclude', 'aga': 'exclude', 'spite': 'replace',
                       'nothingness': 'replace', 'versus': 'exclude',
                       'opposed': 'exclude', 'excluding': 'exclude'}
    before_by = {r['English Word']: r for r in rows}
    after_by = {r['English Word']: r for r in revised}
    for row in revised:
        if row['English Word'] not in actions:
            assert row == before_by[row['English Word']]
        assert all(row[k] == before_by[row['English Word']][k] for k in emotions.KEYS)
    assert after_by['spite']['Russian Word'] == 'злоба'
    assert after_by['nothingness']['Russian Word'] == 'небытие'
    assert len(revised) == len(rows) - 5
    assert all(c['reason'] and c['rule'] for c in changes)
    # Entry guards make an updated/mismatched source fail, rather than patching it silently.
    broken = deepcopy(rows)
    next(r for r in broken if r['English Word'] == 'spite')['Russian Word'] = 'другой перевод'
    try:
        emotions.apply_literary_profile(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched source accepted')
    raw, raw_digest = emotions.load_lexicon(morph, profile='raw')
    edited, digest = emotions.load_lexicon(morph, profile='literary')
    default, _ = emotions.load_lexicon(morph)
    assert default == edited and digest == raw_digest == emotions.SOURCE_SHA256
    assert set.union(*edited.values()) == set(range(8))
    def key(text):
        return tuple(emotions.tokens(text, morph, {}))
    for word in ('что', 'ага', 'несмотря', 'против', 'исключая', 'ничто'):
        assert key(word) in raw and key(word) not in edited, word
    assert edited[key('небытие')] == {1}
    assert edited[key('злоба')] == raw[key('злоба')]
    for word in ('правда', 'конечно', 'случайно', 'иначе', 'несомненно', 'прямо', 'ура'):
        assert raw[key(word)] == edited[key(word)], word
    assert original == emotions.LEXICON_PATH.read_bytes()
    # The generic single-token policy must not erase prepositions inside an expression.
    toy = [{'English Word': 'fixture', 'Russian Word': 'против воли',
            **{k: '1' if k == 'fear' else '0' for k in emotions.KEYS}}]
    complete = rows + toy
    preserved, _ = emotions.apply_literary_profile(complete)
    assert any(r['English Word'] == 'fixture' for r in preserved)
    # Grammar and short prose examples are independent of any particular novel.
    for text in ('Что?', 'Ага.', 'Несмотря.', 'Против.', 'Исключая.', 'Ничто.'):
        assert any(emotions.count_chunks([text], 1, raw, morph)[0]['counts'])
        assert not any(emotions.count_chunks([text], 1, edited, morph)[0]['counts'])
    cases = ['Он сказал, что она вернётся.', 'Он стоял против окна.',
             'Несмотря на дождь, они шли.', 'Она улыбнулась сквозь слёзы.',
             'Правда оказалась горькой.', 'Злоба не покидала его.', 'Ура!']
    raw_chunks = emotions.count_chunks(cases, 2, raw, morph)
    chunks = emotions.count_chunks(cases, 2, edited, morph)
    assert [c['words'] for c in raw_chunks] == [c['words'] for c in chunks]
    for chunk in chunks:
        assert chunk['counts'] == [sum(s['counts'][i] for s in chunk['sentences']) for i in range(8)]
        assert chunk['densities'] == emotions.densities(chunk['counts'], chunk['words'])
        for sentence in chunk['sentences']:
            assert all(m['lemma'] not in ('что', 'против', 'несмотря', 'исключая') for m in sentence['markers'])
    for text in ('Он был счастлив.', 'Он не был счастлив.'):
        a = emotions.count_chunks([text], 1, raw, morph)[0]
        b = emotions.count_chunks([text], 1, edited, morph)[0]
        assert a['counts'] == b['counts'], 'Existing phrase/negation behavior changed'
    simple = {key('счастлив'): {2}}
    positive = emotions.count_chunks(['Он был счастлив.'], 1, simple, morph)[0]
    negative = emotions.count_chunks(['Он не был счастлив.'], 1, simple, morph)[0]
    assert positive['counts'] == negative['counts'], 'Generic negation inference was introduced'
    assert negative['words'] == positive['words'] + 1
    info = emotions.profile_metadata('literary')
    assert info['id'] == 'nrc-ru-literary-v1' and len(info['sha256']) == 64
    assert emotions.profile_metadata('raw')['sha256'] == raw_digest
    with tempfile.TemporaryDirectory() as folder:
        for profile, lexicon in (('raw', raw), ('literary', edited)):
            paths = emotions.write_results(cases, [0.] * len(cases), 2, lexicon,
                                           digest, morph, Path(folder) / profile, profile=profile)
            for kind in ('txt', 'html'):
                report = Path(paths[kind]).read_text(encoding='utf-8')
                assert emotions.profile_metadata(profile)['id'] in report
                assert emotions.profile_metadata(profile)['sha256'] in report
            for kind in ('csv', 'sentences_csv'):
                with open(paths[kind], encoding='utf-8-sig') as f:
                    table = list(csv.reader(f, delimiter=';'))
                assert table[0][-2:] == ['Профиль словаря', 'SHA256 профиля']
                assert all(r[-2] == emotions.profile_metadata(profile)['id'] for r in table[1:])
    print('[smoke-emotion-profile] OK: seven source edits, original retained, general prose/grammar and provenance')


if __name__ == '__main__':
    main()
