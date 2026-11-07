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
from matplotlib.colors import LinearSegmentedColormap

KEYS = ('anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust')
LABELS = ('Гнев', 'Ожидание', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление', 'Доверие')
COLORS = ('#ee8585', '#ffc477', '#df8ad0', '#ad88d4', '#fff080', '#91c4f2', '#8de2e2', '#b3e580')
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


def draw_emotion_chunks(ax, chunks, smoothed):
    """Fill [-1, smoothed tone] within each chunk; tie colors run bottom to top."""
    x = np.arange(1, len(smoothed) + 1)
    for chunk in chunks:
        left, right = chunk['start'] - .5, chunk['end'] + .5
        local_x = np.r_[left, x[(x > left) & (x < right)], right]
        top = np.interp(local_x, x, smoothed)
        leaders = chunk['leaders']
        if not leaders:
            ax.fill_between(local_x, -1, top, facecolor='#eeeeee',
                            hatch='///', edgecolor='#999999', linewidth=0,
                            zorder=1)
        elif len(leaders) == 1:
            ax.fill_between(local_x, -1, top, color=COLORS[leaders[0]],
                            linewidth=0, zorder=1)
        else:
            # Clip an opaque gradient to the same polygon as a solid fill.
            region = ax.fill_between(local_x, -1, top, facecolor='none',
                                     linewidth=0, zorder=1)
            cmap = LinearSegmentedColormap.from_list(
                'tied_emotions', [COLORS[i] for i in leaders], N=256)
            image = ax.imshow(np.linspace(0, 1, 256).reshape(-1, 1),
                              extent=(left, right, -1, max(float(max(top)), -.999999)),
                              origin='lower', aspect='auto', cmap=cmap,
                              interpolation='bilinear', zorder=1)
            image.set_clip_path(region.get_paths()[0], ax.transData)
        ax.axvline(left, color='#777777', linewidth=.5, zorder=2)
    if chunks:
        ax.axvline(chunks[-1]['end'] + .5, color='#777777', linewidth=.5, zorder=2)


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
        x = range(1, len(sentiments) + 1)
        ax.set_facecolor('#f5f5f5')
        ax.plot(x, sentiments, color='blue', linewidth=.7, label='RuSentiLex — тональность фрагментов', zorder=3)
        if len(sentiments):
            smoothed = uniform_filter1d(np.asarray(sentiments, dtype=float), 2 * chunk_size + 1)
            draw_emotion_chunks(ax, chunks, smoothed)
            ax.plot(np.r_[.5, x, len(sentiments) + .5],
                    np.r_[smoothed[0], smoothed, smoothed[-1]],
                    color='black', linewidth=2,
                    label='Скользящее среднее тональности', zorder=4)
            ax.set_xlim(.5, len(sentiments) + .5)
        ax.set_ylim(-1, 1)
        ax.set(title=f'Тональность и ведущие эмоции NRC; фрагмент {chunk_size} предложений', xlabel='Номер предложения', ylabel='Тональность [-1; 1]')
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=color, label=name) for color, name in zip(COLORS, LABELS)]
        handles.append(Patch(facecolor='#eeeeee', hatch='///', label='Маркеры не найдены'))
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.16), ncol=4)
        fig.text(.5, .055,
                 f'Чёрная линия — скользящее среднее тональности, окно {2 * chunk_size + 1} предложений.\n'
                 'Заливка — ведущие эмоции фрагмента; градиент — равные лидеры (снизу вверх в порядке легенды).',
                 ha='center', fontsize=9)
        fig.subplots_adjust(bottom=.33)
        fig.savefig(paths['png'], bbox_inches='tight')
    finally:
        plt.close(fig)
    return paths
