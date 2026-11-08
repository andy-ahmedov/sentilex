"""Desktop NRC EmoLex 2022 markers and word-normalized emotion densities."""
import csv
import hashlib
import io
import json
from html import escape
from pathlib import Path
import re

import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# Preserve the five existing color/index assignments and add NRC's other classes.
KEYS = ('anger', 'sadness', 'joy', 'fear', 'surprise', 'anticipation', 'disgust', 'trust')
LABELS = ('Злость', 'Грусть', 'Радость', 'Страх', 'Удивление', 'Ожидание', 'Отвращение', 'Доверие')
COLORS = ('#ee8585', '#91c4f2', '#fff080', '#ad88d4', '#8de2e2', '#ffc477', '#df8ad0', '#b3e580')
DATA_DIR = Path(__file__).resolve().parent / 'data'
LEXICON_PATH = DATA_DIR / 'nrc2022' / 'Russian-NRC-EmoLex.txt'
SOURCE_URL = 'https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm'
SOURCE_SHA256 = '96849cb99a395e1e749030fa6c115860e2664f7690a1beac82c47485d42ce75a'
SOURCE_ROWS = 14154
PUBLICATION = 'https://doi.org/10.1111/j.1467-8640.2012.00460.x'
DICTIONARY_NAME = 'NRC EmoLex 2022'
NO_MARKERS = 'Эмоциональные слова не найдены'
TOKEN = re.compile(r'[^\W\d_]+', re.UNICODE)


def tokens(text, morph, cache):
    result = []
    for word in TOKEN.findall(text.lower().replace('ё', 'е')):
        if word not in cache:
            cache[word] = morph.parse(word)[0].normal_form.replace('ё', 'е')
        result.append(cache[word])
    return result


def normalize_entries(rows, morph):
    """Union binary emotion labels across normalized Russian translations."""
    lexicon, cache = {}, {}
    for row in rows:
        word = row['Russian Word']
        categories = {i for i, key in enumerate(KEYS) if row[key] == '1'}
        if not categories or not re.search('[а-яё]', word.lower()):
            continue
        key = tuple(tokens(word, morph, cache))
        if key:
            lexicon.setdefault(key, set()).update(categories)
    return lexicon


def load_lexicon(morph):
    try:
        data = LEXICON_PATH.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            'Не найден встроенный NRC EmoLex 2022. Для запуска из исходников '
            'подготовьте локальный ресурс: python tools/prepare_nrc2022.py') from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('Встроенный словарь NRC EmoLex 2022 повреждён или имеет другую версию')
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig')), delimiter='\t'))
    required = {'English Word', 'Russian Word', 'positive', 'negative', *KEYS}
    if (len(rows) != SOURCE_ROWS or set(rows[0]) != required
            or any(row[key] not in ('0', '1') for row in rows for key in KEYS)
            or any(not any(row[key] == '1' for row in rows) for key in KEYS)):
        raise ValueError('Неверный состав встроенного NRC EmoLex 2022')
    return normalize_entries(rows, morph), digest


def source_metadata():
    return json.loads((DATA_DIR / 'NRC2022.SOURCE.json').read_text(encoding='utf-8'))


def densities(counts, words):
    return [value / words if words else 0.0 for value in counts]


def count_chunks(sentences, chunk_size, lexicon, morph):
    if chunk_size < 1:
        raise ValueError('Размер фрагмента должен быть положительным')
    cache, chunks = {}, []
    lengths = sorted({len(k) for k in lexicon}, reverse=True)
    for start in range(0, len(sentences), chunk_size):
        counts, words, details = [0] * len(KEYS), 0, []
        for number, sentence in enumerate(sentences[start:start + chunk_size], start + 1):
            seq = tokens(sentence, morph, cache)
            # Track original character positions through lower-case expansion.
            positions = [i for i, char in enumerate(sentence)
                         for _ in char.lower()]
            spans = list(TOKEN.finditer(sentence.lower().replace('ё', 'е')))
            sentence_counts, markers = [0] * len(KEYS), []
            words += len(seq)
            pos = 0
            while pos < len(seq):
                for size in lengths:
                    categories = lexicon.get(tuple(seq[pos:pos + size]))
                    if categories and pos + size <= len(seq):
                        for category in categories:
                            sentence_counts[category] += 1
                        left = positions[spans[pos].start()]
                        right = positions[spans[pos + size - 1].end() - 1] + 1
                        markers.append(dict(start=left, end=right,
                                            text=sentence[left:right],
                                            lemma=' '.join(seq[pos:pos + size]),
                                            categories=sorted(categories)))
                        pos += size
                        break
                else:
                    pos += 1
            counts = [a + b for a, b in zip(counts, sentence_counts)]
            details.append(dict(number=number, text=sentence, markers=markers,
                                counts=sentence_counts, words=len(seq),
                                densities=densities(sentence_counts, len(seq)),
                                leaders=find_leaders(densities(sentence_counts, len(seq)))))
        end = min(start + chunk_size, len(sentences))
        chunks.append(dict(start=start + 1, end=end, words=words, counts=counts,
                           densities=densities(counts, words),
                           leaders=find_leaders(densities(counts, words)), sentences=details))
    return chunks


def find_leaders(counts):
    maximum = max(counts, default=0)
    return [i for i, value in enumerate(counts) if value == maximum] if maximum else []


def leader_names(leaders):
    return ' / '.join(LABELS[i] for i in leaders) or NO_MARKERS


def count_description(counts):
    return '; '.join(f'{label}: {value}' for label, value in zip(LABELS, counts))


def marker_description(marker, detailed=False):
    lemma = f' [{marker["lemma"]}]' if detailed else ''
    return f'«{marker["text"]}»{lemma} → ' + leader_names(marker['categories'])


def density_description(values):
    return 'Маркеров на 100 слов: ' + '; '.join(
        f'{label}: {100 * value:.2f}' for label, value in zip(LABELS, values))


def leader_description(leaders):
    title = 'Преобладающие эмоции' if len(leaders) > 1 else 'Преобладающая эмоция'
    return title + ': ' + leader_names(leaders)


def sentence_report(chunks):
    """Return TXT and escaped, offline HTML from the actual counting trace."""
    intro = ('Выделены слова из эмоционального словаря. Под каждым предложением '
             'указано, какие эмоции с ними связаны')
    explanation = ('Показан текст после предобработки. Плотность — число маркеров '
                   'категории на все слова; ниже она показана на 100 слов. '
                   'Общий знаменатель позволяет сравнивать фрагменты разной длины, '
                   'но не меняет преобладающую категорию внутри фрагмента. '
                   'Контекст, отрицания и ирония не интерпретируются. '
                   'Отсутствие совпадений не доказывает отсутствие эмоций.')
    lines = [intro, explanation]
    body = [f'<p>{escape(intro)}</p>', '<h1>Эмоциональные кластеры — маркеры</h1>',
            f'<p>{escape(explanation)}</p>']
    for number, chunk in enumerate(chunks, 1):
        title = f'Фрагмент {number}: предложения {chunk["start"]}–{chunk["end"]}'
        leaders = leader_description(chunk['leaders'])
        metrics = (f'Слов: {chunk["words"]}. ' + count_description(chunk['counts'])
                   + '. ' + density_description(chunk['densities']))
        lines.extend(['', title, leaders, metrics])
        body.extend([f'<h2>{escape(title)}</h2>',
                     f'<p><b>{escape(leaders)}</b><br>{escape(metrics)}</p>'])
        for sentence in chunk['sentences']:
            heading = f'Предложение {sentence["number"]}'
            label = leader_description(sentence['leaders'])
            metrics = (f'Слов: {sentence["words"]}. ' + count_description(sentence['counts'])
                       + '. ' + density_description(sentence['densities']))
            lines.extend([heading + ': ' + sentence['text'], label])
            highlighted, pos = [], 0
            for marker in sentence['markers']:
                highlighted.extend([escape(sentence['text'][pos:marker['start']]),
                                    '<span style="background-color:#fff1b8; color:#222222">',
                                    escape(sentence['text'][marker['start']:marker['end']]), '</span>'])
                pos = marker['end']
            highlighted.append(escape(sentence['text'][pos:]))
            body.extend([f'<h3>{heading}</h3>', '<p>' + ''.join(highlighted) + '</p>',
                         f'<p><b>{escape(label)}</b></p>'])
            if sentence['markers']:
                found = 'Найдено: ' + '; '.join(marker_description(m) for m in sentence['markers'])
            else:
                found = 'В этом предложении не найдены слова из словаря'
            lines.extend([found, 'Подробности: ' + metrics])
            lines.extend('  ' + marker_description(m, detailed=True) for m in sentence['markers'])
            body.extend([f'<p>{escape(found)}</p>', '<details><summary>Подробности</summary>',
                         f'<p>{escape(metrics)}</p>', '<ul>',
                         ''.join(f'<li>{escape(marker_description(m, detailed=True))}</li>'
                                 for m in sentence['markers']), '</ul></details>'])
    html = ('<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8">'
            '<title>Эмоциональные кластеры — маркеры</title>'
            '<style>body {font-family:sans-serif; color:#222; background:#fff; margin:24px;} '
            'h2 {color:#483d65;} p,li {line-height:1.5;}</style></head><body>'
            + ''.join(body) + '</body></html>')
    return '\n'.join(lines), html


def summarize(chunks):
    totals = [sum(c['counts'][i] for c in chunks) for i in range(len(KEYS))]
    associations = sum(totals)
    n = sum(c['end'] - c['start'] + 1 for c in chunks)
    shares, unknown = [0.] * len(KEYS), 0
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


def draw_leader_matrix(ax, chunks):
    """Use named rows and exact palette colors to disambiguate gradients."""
    for chunk in chunks:
        left = chunk['start'] - .5
        width = chunk['end'] - chunk['start'] + 1
        if chunk['leaders']:
            for category in chunk['leaders']:
                ax.add_patch(Rectangle((left, category - .5), width, 1,
                                       facecolor=COLORS[category], linewidth=0))
        else:
            ax.add_patch(Rectangle((left, -.5), width, len(KEYS), facecolor='#eeeeee',
                                   edgecolor='#999999', hatch='///', linewidth=0))
        ax.axvline(left, color='#777777', linewidth=.4)
    if chunks:
        ax.axvline(chunks[-1]['end'] + .5, color='#777777', linewidth=.4)
    for y in np.arange(.5, len(KEYS) - .5):
        ax.axhline(y, color='#dddddd', linewidth=.4)
    ax.set_yticks(range(len(KEYS)), LABELS, fontsize=9)
    ax.set_ylim(len(KEYS) - .5, -.5)
    ax.set_title('Преобладающие эмоции по фрагментам', fontsize=10)
    ax.set_xlabel('Номер предложения')


def draw_density(ax, chunks):
    """Five constant-per-chunk curves with shared sentence boundaries."""
    edges = [chunk['start'] - .5 for chunk in chunks]
    if chunks:
        edges.append(chunks[-1]['end'] + .5)
    for category, (label, color) in enumerate(zip(LABELS, COLORS)):
        values = [100 * chunk['densities'][category] for chunk in chunks]
        ax.stairs(values, edges, baseline=None, color=color, linewidth=2, label=label)
    for edge in edges:
        ax.axvline(edge, color='#aaaaaa', linewidth=.4, zorder=0)
    if chunks:
        ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(title='Плотность эмоций', xlabel='Номер предложения',
           ylabel='Маркеров на 100 слов')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=.25)


def result_paths(prefix):
    paths = {k: str(prefix) + '_emotions.' + k for k in ('png', 'txt', 'csv', 'html')}
    paths['density_png'] = str(prefix) + '_emotion_density.png'
    paths['sentences_csv'] = str(prefix) + '_emotion_sentences.csv'
    return paths


def write_results(sentences, sentiments, chunk_size, lexicon, digest, morph, prefix):
    chunks = count_chunks(sentences, chunk_size, lexicon, morph)
    summary = summarize(chunks)
    paths = result_paths(prefix)
    density_headers = [label + ' — плотность (маркеров/слово)' for label in LABELS]
    with open(paths['csv'], 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Фрагмент', 'Начало', 'Конец', 'Предложений', 'Слов']
                        + list(LABELS) + density_headers + ['Преобладающие эмоции'])
        for number, c in enumerate(chunks, 1):
            writer.writerow([number, c['start'], c['end'], c['end'] - c['start'] + 1, c['words']]
                            + c['counts'] + c['densities'] + [leader_names(c['leaders'])])
    with open(paths['sentences_csv'], 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Фрагмент', 'Предложение', 'Текст', 'Слов'] + list(LABELS)
                        + density_headers + ['Преобладающие эмоции', 'Найденные маркеры и категории'])
        for number, c in enumerate(chunks, 1):
            for sentence in c['sentences']:
                writer.writerow([number, sentence['number'], sentence['text'], sentence['words']]
                                + sentence['counts'] + sentence['densities']
                                + [leader_names(sentence['leaders']),
                                   '; '.join(marker_description(m, detailed=True) for m in sentence['markers'])])
    lines = ['Эмоциональные кластеры NRC EmoLex 0.92; русский перевод 2022',
             'Авторы: Saif M. Mohammad, Peter D. Turney; National Research Council Canada.',
             'Английские эмоциональные метки: ручная краудсорсинговая разметка.',
             'Русский перевод: Google Translate, август 2022; не ручная русская разметка.',
             PUBLICATION, SOURCE_URL,
             f'SHA256 русского источника: {digest}',
             f'Записей в исходной русской таблице: {SOURCE_ROWS}',
             f'Размер фрагмента: {chunk_size}',
             'Одно вхождение учитывается один раз в каждой связанной категории.',
             'Словосочетание — один маркер; все его слова входят в число слов.',
             'Метки positive/negative NRC не используются для расчёта тональности RuSentiLex.',
             'Доли маркеров рассчитаны по сумме восьми счётчиков, а не по всем словам.',
             'Доли текста рассчитаны по предложениям; равные категории делят вклад поровну.', '']
    for i, name in enumerate(LABELS):
        lines.append(f'{name}: {summary["totals"][i]}; доля маркеров {summary["marker_shares"][i]:.2%}; доля текста {summary["text_shares"][i]:.2%}')
    lines.append(f'Текст без маркеров: {summary["unknown_share"]:.2%}')
    details, html = sentence_report(chunks)
    # Keep the plain-language introduction first in both formats.
    intro, rest = details.split('\n', 1)
    metadata = '<h2>Сводка и источник</h2><p>' + '<br>'.join(escape(line) for line in lines) + '</p>'
    html = html.replace('</body>', metadata + '</body>', 1)
    Path(paths['html']).write_text(html, encoding='utf-8')
    Path(paths['txt']).write_text(intro + '\n\n' + '\n'.join(lines) + '\n' + rest, encoding='utf-8')
    fig, (ax, matrix) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={'height_ratios': [4, 1.8], 'hspace': .18})
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
                    label='Чёрная линия — сглаженная тональность', zorder=4)
            ax.set_xlim(.5, len(sentiments) + .5)
        ax.set_ylim(-1, 1)
        ax.set(title=f'Тональность и эмоциональные кластеры NRC 2022; размер фрагмента: {chunk_size} предл.', ylabel='Тональность [-1; 1]')
        ax.tick_params(axis='x', labelbottom=False)
        draw_leader_matrix(matrix, chunks)
        handles, labels = ax.get_legend_handles_labels()
        handles += [Patch(facecolor=color, label=name) for color, name in zip(COLORS, LABELS)]
        handles.append(Patch(facecolor='#eeeeee', hatch='///', label=NO_MARKERS))
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .16), ncol=4)
        fig.text(.5, .035,
                 'Градиент — равные преобладающие эмоции. Названия показаны в матрице под кривой.\n'
                 'Штриховка — эмоциональные слова не найдены; границы фрагментов сохранены.',
                 ha='center', fontsize=9)
        fig.subplots_adjust(bottom=.25, left=.11, right=.98)
        fig.savefig(paths['png'], bbox_inches='tight')
    finally:
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        draw_density(ax, chunks)
        fig.text(.5, .01, 'Число маркеров категории / число всех слов × 100; это не доля среди эмоциональных слов.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(paths['density_png'])
    finally:
        plt.close(fig)
    return paths
