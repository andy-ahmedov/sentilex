import matplotlib.pyplot as plt
import csv
from collections import defaultdict
import os
import re
import pymorphy2
from natasha import MorphVocab
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

# Инициализация морфологического анализатора из Natasha
morph_vocab = MorphVocab()

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

def split_into_sentences(text):
    """
    Разбивает текст на предложения, сохраняя разделы и их номера.
    Сохраняет все знаки пунктуации в предложениях.
    """
    sentences = []
    sections = {}
    sentence_counter = 1

    # Улучшенное регулярное выражение для разделения предложений
    sentence_end = re.compile(r'([.!?])(\s+|$)')

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('{') and line.endswith('}'):
            section_name = line[1:-1].strip()
            sections[section_name] = sentence_counter
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

    with open('sections.txt', 'w', encoding='utf-8') as f:
        for section, start in sections.items():
            f.write(f"'{section}' {start}\n")

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

def save_results_to_text(sentences, sentiments, filename="sentiment_data.txt"):
    """
    Сохраняет результаты в простом текстовом файле.

    :param sentiments: Список значений тональности для каждого предложения (список float)
    :param filename: Имя файла для сохранения (строка)
    """
    with open(filename, "w", encoding="utf-8") as f:
        for i, sentiment in enumerate(sentiments):
            f.write(f"{sentiment:+.2f} {sentences[i]}\n")
            

def plot_sentiment_curve(sentiments, sentences, section_positions, section_labels, chunk_size, output_image="sentiment_curve.png"):
    window_size = 2*chunk_size + 1
    polynom_order = 0

    smoothed_savgol = savgol_filter(sentiments, window_size, polynom_order, mode='interp')
    smoothed_uniform = uniform_filter1d(sentiments, window_size)
    smoothed_conv = np.convolve(sentiments, np.ones(window_size)/window_size, mode='same')

    x = range(1, len(sentences) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(x, sentiments, label="RuSentiLex", linewidth=0.5, color="blue")
    plt.plot(x, smoothed_savgol, label="Фильтр Савицкого-Голея", linewidth=1.5, color="red")
    plt.plot(x, smoothed_uniform, label="Скользящее среднее", linewidth=1.5, color="green")
    plt.plot(x, smoothed_conv, label="Сглаживание", linewidth=1.5, color="magenta")

    if section_positions:
        for position in section_positions:
            plt.axvline(x=position, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        plt.xticks(section_positions, section_labels, rotation=45, fontsize=10)

    plt.grid(which='major', axis='y', linestyle='-', linewidth=0.5, color='gray', alpha=0.7)

    plt.title('Кривые эмоциональной тональности', fontsize=14)
    plt.xlabel('Главы' if section_positions else 'Предложения', fontsize=12)
    plt.ylabel('Оценка тональности [-1;1]', fontsize=12)
    plt.legend(fontsize=12)

    plt.savefig(output_image)
    plt.close()


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
            