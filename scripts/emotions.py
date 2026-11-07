"""Optional desktop NRC EmoLex analysis; dictionary data stays external."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

KEYS = ('anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust')
LABELS = ('Гнев', 'Ожидание', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление', 'Доверие')
COLORS = ('#e15759', '#f28e2b', '#79706e', '#b07aa1', '#edc948', '#4e79a7', '#76b7b2', '#59a14f')
TOKEN = re.compile(r'[^\W\d_]+', re.UNICODE)


def tokens(text, morph, cache):
    result = []
    for word in TOKEN.findall(text.lower().replace('ё', 'е')):
        if word not in cache:
            cache[word] = morph.parse(word)[0].normal_form.replace('ё', 'е')
        result.append(cache[word])
    return result


def load_lexicon(path, morph):
    from openpyxl import load_workbook
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if Path(path).suffix.lower() == '.zip':
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [n for n in archive.namelist() if n.endswith('NRC-Emotion-Lexicon-v0.92-In105Languages-Nov2017Translations.xlsx') and '/Older Versions/' not in n and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('В архиве не найдена однозначная таблица NRC EmoLex 2017')
            data = archive.read(names[0])
    book = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        rows = book.worksheets[0].iter_rows(values_only=True)
        header = next(rows)
        russian = header.index('Russian (ru)')
        columns = [header.index(k.title()) for k in KEYS]
        lexicon, cache = {}, {}
        for row in rows:
            word = row[russian]
            if not isinstance(word, str) or not re.search('[а-яё]', word.lower()):
                continue
            categories = {i for i, col in enumerate(columns) if row[col] == 1}
            key = tuple(tokens(word, morph, cache))
            if key and categories:
                lexicon.setdefault(key, set()).update(categories)
        if not lexicon:
            raise ValueError('Таблица не содержит русских эмоциональных маркеров')
        return lexicon, digest
    finally:
        book.close()


def count_chunks(sentences, chunk_size, lexicon, morph):
    if chunk_size < 1:
        raise ValueError('Размер фрагмента должен быть положительным')
    cache, chunks = {}, []
    lengths = sorted({len(k) for k in lexicon}, reverse=True)
    for start in range(0, len(sentences), chunk_size):
        counts, words = [0] * 8, 0
        for sentence in sentences[start:start + chunk_size]:
            seq = tokens(sentence, morph, cache)
            words += len(seq)
            pos = 0
            while pos < len(seq):
                for size in lengths:
                    categories = lexicon.get(tuple(seq[pos:pos + size]))
                    if categories and pos + size <= len(seq):
                        for category in categories:
                            counts[category] += 1
                        pos += size
                        break
                else:
                    pos += 1
        end = min(start + chunk_size, len(sentences))
        maximum = max(counts)
        leaders = [i for i, n in enumerate(counts) if n == maximum] if maximum else []
        chunks.append(dict(start=start + 1, end=end, words=words, counts=counts, leaders=leaders))
    return chunks


def summarize(chunks):
    totals = [sum(c['counts'][i] for c in chunks) for i in range(8)]
    associations = sum(totals)
    n = sum(c['end'] - c['start'] + 1 for c in chunks)
    shares, unknown = [0.] * 8, 0
    for c in chunks:
        length = c['end'] - c['start'] + 1
        if c['leaders']:
            for i in c['leaders']:
                shares[i] += length / len(c['leaders'])
        else:
            unknown += length
    return dict(totals=totals, marker_shares=[v / associations if associations else 0 for v in totals], text_shares=[v / n if n else 0 for v in shares], unknown_share=unknown / n if n else 0)


def write_results(sentences, sentiments, chunk_size, lexicon, digest, morph, prefix):
    chunks = count_chunks(sentences, chunk_size, lexicon, morph)
    summary = summarize(chunks)
    paths = {k: str(prefix) + '_emotions.' + k for k in ('png', 'txt', 'csv')}
    with open(paths['csv'], 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Фрагмент', 'Начало', 'Конец', 'Предложений', 'Слов'] + list(LABELS) + ['Лидеры'])
        for number, c in enumerate(chunks, 1):
            writer.writerow([number, c['start'], c['end'], c['end'] - c['start'] + 1, c['words']] + c['counts'] + [' / '.join(LABELS[i] for i in c['leaders']) or 'Маркеры не найдены'])
    lines = ['Эмоциональные маркеры NRC EmoLex 0.92; автоматический русский перевод 2017', 'Авторы: Saif M. Mohammad, Peter D. Turney; NRC Canada', 'https://www.saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm', f'SHA256 источника: {digest}', f'Размер фрагмента: {chunk_size}', 'Лексические ассоциации; контекст и отрицания не интерпретируются.', 'Слово с несколькими метками учитывается в каждой категории.', 'Доли текста рассчитаны по предложениям; равные лидеры делят вклад поровну.', '']
    for i, name in enumerate(LABELS):
        lines.append(f'{name}: {summary["totals"][i]}; доля маркеров {summary["marker_shares"][i]:.2%}; доля текста {summary["text_shares"][i]:.2%}')
    lines.append(f'Текст без маркеров: {summary["unknown_share"]:.2%}')
    lines.extend(['', 'Подсчёты по фрагментам:', Path(paths['csv']).read_text(encoding='utf-8-sig')])
    Path(paths['txt']).write_text('\n'.join(lines), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(14, 8))
    try:
        for c in chunks:
            leaders = c['leaders']
            for band, i in enumerate(leaders):
                ax.axvspan(c['start'] - .5, c['end'] + .5, ymin=band / len(leaders), ymax=(band + 1) / len(leaders), color=COLORS[i], alpha=.24)
            if not leaders:
                ax.axvspan(c['start'] - .5, c['end'] + .5, facecolor='#eeeeee', hatch='///', edgecolor='#999999', alpha=.4)
            ax.axvline(c['start'] - .5, color='black', linewidth=.6)
        if chunks:
            ax.axvline(chunks[-1]['end'] + .5, color='black', linewidth=.6)
        x = range(1, len(sentiments) + 1)
        ax.plot(x, sentiments, color='blue', linewidth=.5, label='RuSentiLex')
        if len(sentiments):
            ax.plot(x, uniform_filter1d(np.asarray(sentiments, dtype=float), 2 * chunk_size + 1), color='black', linewidth=1.5, label='Скользящее среднее')
        ax.set(title=f'Тональность и ведущие эмоции NRC; фрагмент {chunk_size} предложений', xlabel='Номер предложения', ylabel='Тональность [-1; 1]')
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=color, label=name) for color, name in zip(COLORS, LABELS)]
        handles.append(Patch(facecolor='#eeeeee', hatch='///', label='Маркеры не найдены'))
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.13), ncol=4)
        fig.subplots_adjust(bottom=.28)
        fig.savefig(paths['png'], bbox_inches='tight')
    finally:
        plt.close(fig)
    return paths
