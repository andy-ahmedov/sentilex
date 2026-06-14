import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import os
import re
import pymorphy2
from natasha import MorphVocab
import numpy as np
from scipy.ndimage import uniform_filter1d

# Инициализация морфологического анализатора из Natasha
morph_vocab = MorphVocab()

METRIC_DISPLAY_NAMES = (
    ("volatility_amplitude", "Волатильность"),
    ("path_length", "Общая изменчивость"),
    ("mean_change", "Удельная изменчивость"),
)

# Функция для загрузки RuSentiLex
def load_rusentilex(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден. Пожалуйста, загрузите файл в среду.")

    lexicon = {}
    phrase_lexicon = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Убираем лишние символы и пробелы
            line = line.strip()

            # Пропускаем пустые строки и комментарии
            if not line or line.startswith('!') or line.startswith('[page'):
                continue

            # Разделяем строку по запятым
            parts = re.split(r',\s*', line)

            if len(parts) >= 4:
                word_or_phrase = parts[0]  # Слово или словосочетание
                lemma = parts[2]  # Лемматизированная форма
                sentiment = parts[3]  # Тональность

                # Определяем числовое значение тональности
                if sentiment == "positive":
                    score = 1.0
                elif sentiment == "negative":
                    score = -1.0
                elif sentiment == "neutral":
                    score = 0.0
                elif sentiment == "positive/negative":
                    score = 0.5  # Положительная/отрицательная — присваиваем нейтральное значение
                else:
                    score = 0.0  # Если тональность не определена, считаем её нейтральной

                # Добавляем в словарь слов или словосочетаний
                if ' ' in lemma:  # Если лемма содержит пробел, это словосочетание
                    phrase_lexicon[lemma] = score
                else:  # Иначе это отдельное слово
                    lexicon[lemma] = score

    return lexicon, phrase_lexicon


def split_sentence(sentence):
    # Паттерн для разделителей (знаки препинания с пробелами)
    split_pattern = r'''
        \s*                                  # Пробелы перед разделителем
        (?:                                  # Группа без захвата:
            [.,!?;()\[\]{}—]                # Стандартные разделители
            |                               # ИЛИ
            (?<!\w)-(?!\w)                  # Дефис не внутри слова
            |                               # ИЛИ
            (?<!\d):(?!\d)                  # Двоеточие не между цифрами
        )+                                   # Один или более разделителей
        \s*                                  # Пробелы после разделителя
    '''

    # Разделение на фрагменты
    parts = re.split(split_pattern, sentence, flags=re.VERBOSE | re.IGNORECASE)

    # Очистка фрагментов
    cleaned = []
    for part in parts:
        if not part or part.isspace():
            continue
        part = part.strip()
        # Удаляем краевые не-словные символы (кроме дефиса)
        cleaned_part = re.sub(r'^[^\wа-яА-ЯёЁ-]+', '', part)
        cleaned_part = re.sub(r'[^\wа-яА-ЯёЁ-]+$', '', cleaned_part)
        # Удаляем дефисы, за которыми НЕТ букв/цифр (включая конец строки)
        cleaned_part = re.sub(r'-(?!\w)', '', cleaned_part)
        # Удаляем дефисы, перед которыми НЕТ букв/цифр (начало строки)
        cleaned_part = re.sub(r'(?<!\w)-', '', cleaned_part)
        if cleaned_part:
            cleaned.append(cleaned_part)

    return cleaned

def normalize_word(word):
    """
    Приводит слово к нормальной форме (лемме) и заменяет 'ё' на 'е'.

    :param word: Исходное слово (строка)
    :return: Нормализованное слово (строка)
    """
    parsed_word = morph_vocab.parse(word)[0]  # Лемматизируем слово
    normal_form = parsed_word.normal_form  # Получаем нормальную форму
    return normal_form.replace('ё', 'е')  # Заменяем 'ё' на 'е'

def normalize_phrase(phrase):
    """
    Нормализует фразу (приводит каждое слово к нормальной форме и заменяет 'ё' на 'е').

    :param phrase: Исходная фраза (строка)
    :return: Нормализованная фраза (строка)
    """
    words = phrase.split()  # Разбиваем фразу на слова
    normalized_words = [normalize_word(word) for word in words]  # Нормализуем каждое слово
    return ' '.join(normalized_words)  # Собираем обратно в фразу

def lemmatize_phrase(phrase):
    """
    Лемматизирует словосочетание.

    :param phrase: Словосочетание (строка)
    :return: Лемматизированное словосочетание (строка)
    """
    return ' '.join(morph_vocab.parse(word)[0].normal_form for word in phrase.split())

def find_lemmatized_phrases(sentence, phrase_lexicon):

    phrases_found = []

    # Разбиваем предложение на фрагменты
    fragments = split_sentence(sentence)
    #print(fragments)

    # Ищем словосочетания в каждом фрагменте
    for fragment in fragments:
        words = re.sub(r'[^\w\s]', '', fragment).lower().split()  # Очистка от знаков препинания

        for i in range(len(words)):
            for j in range(i + 2, min(i + 6, len(words) + 1)):  # Максимальная длина словосочетания: 5 слов
                phrase = ' '.join(words[i:j])
                lemma = normalize_phrase(lemmatize_phrase(phrase))
                #print(lemma)

                if lemma in phrase_lexicon:
                    phrases_found.append((phrase, phrase_lexicon[lemma]))

    return phrases_found

def calculate_sentiment(sentence, lexicon, phrase_lexicon):
    """
    Вычисляет тональность предложения.

    :param sentence: Исходное предложение (строка)
    :param lexicon: Словарь слов (ключ: слово, значение: тональность)
    :param phrase_lexicon: Словарь словосочетаний (ключ: словосочетание, значение: тональность)
    :return: Средняя тональность предложения
    """
    #print(f"Обработка предложения: {sentence}")

    # Разбиваем предложение на фрагменты
    fragments = split_sentence(sentence)
    #print("Фрагменты:", fragments)

    sentiment_score = 0.0
    word_count = 0

    # Обрабатываем каждый фрагмент
    for fragment in fragments:
        #print(f"Фрагмент: {fragment}")
        # Ищем лемматизированные словосочетания
        phrases = find_lemmatized_phrases(fragment, phrase_lexicon)
        for phrase, score in phrases:
            sentiment_score += score
            word_count += 1
            #print(f"Словосочетание: {phrase} ({lemmatize_phrase(phrase)}), Тональность: {score}")

        # Удаляем найденные словосочетания из фрагмента
        for phrase, _ in phrases:
            fragment = re.sub(re.escape(phrase), '', fragment)

        # Разбиваем оставшуюся часть фрагмента на слова
        words = re.findall(r'\b\w+\b', fragment.lower())
        #print(words)

        negate = False
        for word in words:
            lemma = morph_vocab.parse(word)[0].normal_form
            #print(f"word: {word}, lemma: {lemma}")

            # Проверяем отрицание
            if word in {"не", "ни", "никак", "нисколько"}:
                negate = True
                continue

            if lemma in lexicon:
                sentiment = lexicon[lemma]
                if negate and sentiment != 0:  # Инвертируем тональность при отрицании
                    sentiment = -sentiment
                sentiment_score += sentiment
                word_count += 1
                #print(f"Слово: {word} ({lemma}), Тональность: {sentiment}")

            # Сброс флага отрицания после обработки слова
            if negate and word not in {"не", "ни", "никак", "нисколько"}:
                negate = False

    # Возвращаем среднюю тональность, если были найдены совпадения
    return sentiment_score / word_count if word_count > 0 else 0

def preprocess_text(text):
    """
    Предварительная обработка текста с сохранением разделов в отдельных строках.
    Сохраняет дефисы, тире, кавычки и другие важные символы.
    """
    # Добавляем разрешенные символы: - (дефис), — (тире), "", «», {}
    allowed_symbols = r'\-\—\«\»\"\{\}'

    lines = text.split('\n')
    processed_lines = []

    for line in lines:
        # Сохраняем структуру разделов {Раздел}
        if line.strip().startswith('{') and line.strip().endswith('}'):
            processed_lines.append(line.strip())
            continue

        # Основная обработка:
        # 1. Заменяем множественные пробелы на один
        line = re.sub(r'\s+', ' ', line.strip())

        # 2. Удаляем нежелательные символы, сохраняя разрешенные
        line = re.sub(
            fr'[^а-яА-ЯёЁa-zA-Z0-9\s.,!?{allowed_symbols}]',
            '',
            line,
            flags=re.IGNORECASE
        )

        processed_lines.append(line)

    return '\n'.join(processed_lines)

def extract_sentences_and_sections(text):
    """
    Разбивает текст на предложения и возвращает позиции разделов без записи на диск.
    Сохраняет все знаки пунктуации в предложениях.
    """
    sentences = []
    sections = []
    sentence_counter = 1

    # Улучшенное регулярное выражение для разделения предложений
    sentence_end = re.compile(r'([.!?])(\s+|$)')

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('{') and line.endswith('}'):
            section_name = line[1:-1].strip()
            sections.append((section_name, sentence_counter))
        else:
            # Разделяем предложения, сохраняя знаки пунктуации
            parts = sentence_end.split(line)
            buffer = []

            for part in parts:
                buffer.append(part)
                if sentence_end.search(part):
                    sent = ''.join(buffer).strip()
                    if sent:
                        sentences.append(sent)
                        sentence_counter += 1
                    buffer = []

            if buffer:
                sent = ''.join(buffer).strip()
                if sent:
                    sentences.append(sent)
                    sentence_counter += 1

    return sentences, sections


def write_sections_file(sections, file_path='sections.txt'):
    with open(file_path, 'w', encoding='utf-8') as f:
        for section, start in sections:
            f.write(f"'{section}' {start}\n")


def split_into_sentences(text):
    """
    Разбивает текст на предложения, сохраняя разделы и их номера.
    Сохраняет все знаки пунктуации в предложениях.
    """
    sentences, sections = extract_sentences_and_sections(text)

    write_sections_file(sections)

    return sentences

# Функция для чтения данных из файла sections.txt
def read_sections(file_path):
    """
    Читает данные из файла sections.txt и возвращает список позиций и меток разделов.
    :param file_path: Путь к файлу sections.txt.
    :return: Список позиций (номеров предложений) и список меток разделов.
    """
    positions = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # Убираем лишние пробелы и символы новой строки

            # Игнорируем пустые строки
            if not line:
                continue

            # Разделяем строку по табуляции
            parts = line.split('\t')

            # Проверяем, что строка содержит ровно два элемента (метка и позиция)
            if len(parts) != 2:
                print(f"Пропущена строка с некорректным форматом: {line}")
                continue

            section, position = parts

            # Убираем кавычки из метки раздела
            label = re.sub(r"^'(.+)'$", r'\1', section)

            # Добавляем метку и позицию в соответствующие списки
            labels.append(label)
            positions.append(int(position))

    return positions, labels


def format_metric_lines(metrics):
    return [
        f"{label}: {metrics[key]:.6f}"
        for key, label in METRIC_DISPLAY_NAMES
    ]


def annotate_extreme_point(ax, x_value, y_value, color, y_offset):
    ax.scatter(x_value, y_value, color=color, s=40, zorder=3)
    ax.annotate(
        f"{y_value:.3f}",
        xy=(x_value, y_value),
        xytext=(10, y_offset),
        textcoords="offset points",
        fontsize=9,
        color=color,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


def highlight_mean_tick(ax, mean_value):
    y_ticks = list(ax.get_yticks())
    if not any(abs(tick - mean_value) < 0.001 for tick in y_ticks):
        y_ticks.append(mean_value)
        y_ticks.sort()

    y_labels = []
    for tick in y_ticks:
        if abs(tick - mean_value) < 0.001:
            y_labels.append(f"{tick:.3f}")
        else:
            y_labels.append(f"{tick:.2f}")

    ax.set_yticks(y_ticks, y_labels, fontsize=9)
    for tick in ax.yaxis.get_major_ticks():
        if abs(tick.get_loc() - mean_value) < 0.001:
            tick.tick1line.set_color("brown")
            tick.tick1line.set_linewidth(2.5)
            tick.tick2line.set_color("brown")
            tick.tick2line.set_linewidth(2.5)
            tick.label1.set_color("brown")
            tick.label1.set_weight("bold")
            tick.label1.set_fontsize(10)


def save_results_to_text(sentences, sentiments, filename="sentiment_data.txt", metrics=None):
    """
    Сохраняет результаты в простом текстовом файле.

    :param sentiments: Список значений тональности для каждого предложения (список float)
    :param filename: Имя файла для сохранения (строка)
    """
    with open(filename, "w", encoding="utf-8") as f:
        for i, sentiment in enumerate(sentiments):
            f.write(f"{sentiment:+.2f} {sentences[i]}\n")

        if metrics:
            f.write("\n")
            f.write("Метрики графика:\n")
            for line in format_metric_lines(metrics):
                f.write(f"{line}\n")
            

def plot_sentiment_curve(
    sentiments,
    sentences,
    section_positions,
    section_labels,
    chunk_size,
    output_image="sentiment_curve.png",
    title_note=None,
):
    window_size = 2 * chunk_size + 1
    smoothed_uniform = uniform_filter1d(sentiments, window_size)

    volatility_amplitude = float(np.std(smoothed_uniform))
    path_length = float(np.sum(np.abs(np.diff(smoothed_uniform))))
    mean_change = path_length / len(smoothed_uniform) if len(smoothed_uniform) > 0 else 0.0

    x = range(1, len(sentences) + 1)

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.plot(x, sentiments, label="RuSentiLex", linewidth=0.5, color="blue")
    ax.plot(x, smoothed_uniform, label="Скользящее среднее", linewidth=1.5, color="green")

    max_index = int(np.argmax(smoothed_uniform))
    min_index = int(np.argmin(smoothed_uniform))
    max_value = float(smoothed_uniform[max_index])
    min_value = float(smoothed_uniform[min_index])
    annotate_extreme_point(ax, max_index + 1, max_value, "red", 10)
    annotate_extreme_point(ax, min_index + 1, min_value, "blue", -15)
    ax.axhline(y=max_value, color="red", linestyle="-", linewidth=2, alpha=0.8)
    ax.axhline(y=min_value, color="blue", linestyle="-", linewidth=2, alpha=0.8)

    max_index_original = int(np.argmax(sentiments))
    min_index_original = int(np.argmin(sentiments))
    max_value_original = float(sentiments[max_index_original])
    min_value_original = float(sentiments[min_index_original])
    annotate_extreme_point(ax, max_index_original + 1, max_value_original, "darkred", 10)
    annotate_extreme_point(ax, min_index_original + 1, min_value_original, "darkblue", -15)
    ax.axhline(y=max_value_original, color="darkred", linestyle="--", linewidth=2, alpha=0.8)
    ax.axhline(y=min_value_original, color="darkblue", linestyle="--", linewidth=2, alpha=0.8)

    mean_value = float(np.mean(smoothed_uniform))
    upper_bound = mean_value + volatility_amplitude
    lower_bound = mean_value - volatility_amplitude
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.2, color="gray")
    ax.axhline(y=upper_bound, color="brown", linestyle="--", linewidth=2, alpha=0.7)
    ax.axhline(y=lower_bound, color="brown", linestyle="--", linewidth=2, alpha=0.7)
    ax.axhline(y=mean_value, color="brown", linestyle="-", linewidth=2, alpha=0.8)
    highlight_mean_tick(ax, mean_value)

    if section_positions:
        for position in section_positions:
            ax.axvline(x=position, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xticks(section_positions, section_labels, rotation=45, fontsize=10)

    ax.grid(which='major', axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)

    title_suffix = title_note or f"chunk_size = {chunk_size}"
    ax.set_title(f'Кривые эмоциональной тональности ({title_suffix})', fontsize=14)
    ax.set_xlabel('Главы' if section_positions else 'Предложения', fontsize=12, labelpad=12)
    ax.set_ylabel('Оценка тональности [-1;1]', fontsize=12)
    ax.legend(fontsize=12)

    metrics_text = "    ".join(format_metric_lines({
        "volatility_amplitude": volatility_amplitude,
        "path_length": path_length,
        "mean_change": mean_change,
    }))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.28)
    fig.text(0.5, 0.06, metrics_text, ha='center', va='bottom', fontsize=10)

    fig.savefig(output_image, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)

    return {
        "volatility_amplitude": volatility_amplitude,
        "path_length": path_length,
        "mean_change": mean_change,
    }


def calculate_sentences_sentiments(sentences, section_positions, chunk_size, lexicon, phrase_lexicon):
    """
    Вычисляет тональность для предложений с группировкой в чанки и учетом разделов

    Параметры:
    sentences (list): Список предложений
    section_positions (list): Позиции начала разделов (1-based)
    sentence_chunk (int): Размер чанка
    lexicon: Словарь для анализа тональности
    phrase_lexicon: Фразовый словарь для анализа

    Возвращает:
    list: Список значений тональности для каждого предложения
    """
    sentiments = []
    current_idx = 0
    section_ptr = 0
    n = len(sentences)
    section_n = len(section_positions)

    while current_idx < n:
        max_chunk_end = current_idx + chunk_size - 1
        chunk_end = min(max_chunk_end, n - 1)

        # Обработка разделов
        while section_ptr < section_n:
            section_start = section_positions[section_ptr] - 1  # Конвертация в 0-based

            if section_start > chunk_end:
                break  # Раздел за пределами текущего чанка

            if section_start >= current_idx:
                if section_start == current_idx:
                    # Пропуск раздела в начале чанка
                    section_ptr += 1
                    current_idx += 1
                    continue
                else:
                    # Обрезка чанка перед разделом
                    chunk_end = section_start - 1
                    break
            section_ptr += 1

        # Корректировка границ чанка
        chunk_end = max(current_idx, min(chunk_end, n - 1))

        # Вычисление тональности для чанка
        chunk = sentences[current_idx:chunk_end + 1]
        if chunk:
            chunk_sentiments = [calculate_sentiment(s, lexicon, phrase_lexicon) for s in chunk]
            avg_sentiment = sum(chunk_sentiments) / len(chunk_sentiments)
            sentiments.extend([avg_sentiment] * len(chunk))

        current_idx = chunk_end + 1

    return sentiments
            
