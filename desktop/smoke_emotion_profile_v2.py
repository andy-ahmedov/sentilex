"""Source-sense-based v2 regression; no novel or corpus frequencies are used."""
import csv
import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from scripts import emotions


def must_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Mismatched source/profile accepted')


def main():
    morph = app.functions.morph_vocab
    original = emotions.LEXICON_PATH.read_bytes()
    rows = list(csv.DictReader(original.decode().splitlines(), delimiter='\t'))
    before = deepcopy(rows)
    v1, old_changes = emotions.apply_literary_profile(rows, profile='literary_v1')
    v2, changes = emotions.apply_literary_profile(rows, profile='literary')
    assert rows == before
    assert len(old_changes) == 7 and len(changes) == 28
    extra = [c for c in changes if c['profile_id'] == 'nrc-ru-literary-v2']
    assert len(extra) == 21 and sum(c['action'] == 'exclude' for c in extra) == 4
    originals = {r['English Word']: r for r in rows}
    edited_ids = {c['english'] for c in changes}
    for row in v2:
        assert all(row[k] == originals[row['English Word']][k] for k in emotions.KEYS)
        if row['English Word'] not in edited_ids:
            assert row == originals[row['English Word']]
    for retained in ('blue', 'labor', 'honest', 'intelligence'):
        assert next(r for r in v2 if r['English Word'] == retained) == originals[retained]
    assert [r['Russian Word'] for r in v2 if r['English Word'] == 'suddenly'] == ['вдруг', 'внезапно']
    for english in ('cross', 'execution', 'tender', 'treat'):
        assert any(r['English Word'] == english for r in v1)
        assert not any(r['English Word'] == english for r in v2)
    # Guards cover changed text, labels, missing and duplicate entry identities.
    for field, value in (('Russian Word', 'другой перевод'), ('sadness', '0')):
        broken = deepcopy(rows)
        next(r for r in broken if r['English Word'] == 'pine')[field] = value
        must_reject(lambda: emotions.apply_literary_profile(broken))
    must_reject(lambda: emotions.apply_literary_profile([r for r in rows if r['English Word'] != 'pine']))
    must_reject(lambda: emotions.apply_literary_profile(rows + [originals['pine']]))
    must_reject(lambda: emotions.read_literary_profile('unknown'))
    # Changing v1 cannot silently change the cumulative v2 profile.
    real_metadata = emotions.profile_metadata
    def altered_parent(profile):
        return {**real_metadata(profile), 'sha256': '0' * 64}
    with patch.object(emotions, 'profile_metadata', side_effect=altered_parent):
        must_reject(emotions.read_literary_profile)
    raw, _ = emotions.load_lexicon(morph, profile='raw')
    first, _ = emotions.load_lexicon(morph, profile='literary_v1')
    second, _ = emotions.load_lexicon(morph, profile='literary')
    assert len(raw) == 3707 and len(first) == 3702
    assert emotions.profile_metadata('literary_v1')['sha256'] == 'cbd2607a69f63549552956dd71ffb56f2ba99b76531646438499edae274a29e7'
    assert emotions.profile_metadata('literary')['id'] == 'nrc-ru-literary-v2'
    assert emotions.profile_metadata('literary')['sha256'] == '119bc130f86078a77053ddf1add6445cfd956e8ceac685fba0c2a9f95b311f6a'
    def key(word):
        return tuple(emotions.tokens(word, morph, {}))
    for word in ('сосна', 'улица', 'содержание', 'корт', 'фунт', 'валовой', 'тендер'):
        assert key(word) in first and key(word) not in second, word
    for word in ('чахнуть', 'сожалеть', 'довольный', 'суд', 'колотить', 'отвратительный', 'вдруг', 'внезапно'):
        assert key(word) in second, word
    for text in ('Вдруг.', 'Внезапно.'):
        trace = emotions.count_chunks([text], 1, second, morph)[0]
        assert trace['counts'][4] == 1
    trace = emotions.count_chunks(['Вдруг внезапно.'], 1, second, morph)[0]
    assert trace['counts'][4] == 2
    # Duplicates from alternative translations union labels, not occurrence counts.
    revised = deepcopy(v2)
    revised.append(next(r for r in v2 if r['English Word'] == 'suddenly'))
    assert emotions.normalize_entries(revised, morph) == second
    assert first[key('синий')] == second[key('синий')]
    samples = ('За окном росла сосна.', 'Письмо описывало содержание договора.',
               'Игра продолжилась на корте.', 'Кабинет был пуст.', 'Валовой доход вырос.')
    removed = {key(w)[0] for w in ('сосна', 'содержание', 'корт', 'кабинет', 'валовой')}
    for text in samples:
        chunk = emotions.count_chunks([text], 1, second, morph)[0]
        assert chunk['words'] == emotions.count_chunks([text], 1, first, morph)[0]['words']
        assert not any(m['lemma'] in removed for s in chunk['sentences'] for m in s['markers'])
    for text in ('Он чахнул в разлуке.', 'Она сожалела об утрате.', 'Собрался кабинет министров.'):
        chunk = emotions.count_chunks([text], 1, second, morph)[0]
        assert chunk['sentences'][0]['markers']
    # Literal source color associations remain; ambiguity is documented, not hidden.
    assert ('гора',) in second  # Current morphology can conflate гора/горе.
    assert hashlib.sha256(emotions.LEXICON_PATH.read_bytes()).hexdigest() == emotions.SOURCE_SHA256
    assert original == emotions.LEXICON_PATH.read_bytes()
    print('[smoke-profile-v2] OK: 21 added decisions, immutable v1/raw, guards, alternatives and independent prose')


if __name__ == '__main__':
    main()
