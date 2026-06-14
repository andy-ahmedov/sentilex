from __future__ import annotations

import base64
from pathlib import Path
import zlib


EXPORT_SCRIPT_TEMPLATE = r'''#!/usr/bin/env python3
"""Standalone SentiSoft analysis script exported from SentiSoft.exe."""

from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
import re
import zlib

import matplotlib.pyplot as plt
from natasha import MorphVocab
import numpy as np
from scipy.ndimage import uniform_filter1d

SCRIPT_MARKER = "SENTILEX_EXPORT_SCRIPT"
LEXICON_PAYLOAD = "__LEXICON_BASE64__"
DEFAULT_CHUNK_SIZE = __DEFAULT_CHUNK_SIZE__

# Plot customization block. Edit these values to adjust the exported figure.
RAW_LINE_COLOR = "blue"
RAW_LINE_WIDTH = 0.5
SMOOTH_LINE_COLOR = "green"
SMOOTH_LINE_WIDTH = 1.5
SECTION_LINE_COLOR = "gray"
SECTION_LINE_STYLE = "--"
SECTION_LINE_WIDTH = 1.0
SECTION_LINE_ALPHA = 0.7
SHOW_ONLY_EVEN_SECTION_LABELS = False
CUSTOM_TITLE_NOTE = None
CUSTOM_PLOT_TITLE = None
CUSTOM_X_LABEL = None
CUSTOM_Y_LABEL = "Оценка тональности [-1;1]"

METRIC_DISPLAY_NAMES = (
    ("volatility_amplitude", "Волатильность"),
    ("path_length", "Общая изменчивость"),
    ("mean_change", "Удельная изменчивость"),
)

morph_vocab = MorphVocab()


def load_embedded_rusentilex():
    payload = zlib.decompress(base64.b64decode(LEXICON_PAYLOAD)).decode("utf-8")
    lexicon = {}
    phrase_lexicon = {}

    for line in payload.splitlines():
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("[page"):
            continue

        parts = re.split(r",\s*", line)
        if len(parts) < 4:
            continue

        lemma = parts[2]
        sentiment = parts[3]
        if sentiment == "positive":
            score = 1.0
        elif sentiment == "negative":
            score = -1.0
        elif sentiment == "neutral":
            score = 0.0
        elif sentiment == "positive/negative":
            score = 0.5
        else:
            score = 0.0

        if " " in lemma:
            phrase_lexicon[lemma] = score
        else:
            lexicon[lemma] = score

    return lexicon, phrase_lexicon


def split_sentence(sentence):
    split_pattern = r"""
        \s*
        (?:
            [.,!?;()\[\]{}—]
            |
            (?<!\w)-(?!\w)
            |
            (?<!\d):(?!\d)
        )+
        \s*
    """

    parts = re.split(split_pattern, sentence, flags=re.VERBOSE | re.IGNORECASE)

    cleaned = []
    for part in parts:
        if not part or part.isspace():
            continue
        part = part.strip()
        cleaned_part = re.sub(r"^[^\wа-яА-ЯёЁ-]+", "", part)
        cleaned_part = re.sub(r"[^\wа-яА-ЯёЁ-]+$", "", cleaned_part)
        cleaned_part = re.sub(r"-(?!\w)", "", cleaned_part)
        cleaned_part = re.sub(r"(?<!\w)-", "", cleaned_part)
        if cleaned_part:
            cleaned.append(cleaned_part)

    return cleaned


def normalize_word(word):
    parsed_word = morph_vocab.parse(word)[0]
    normal_form = parsed_word.normal_form
    return normal_form.replace("ё", "е")


def normalize_phrase(phrase):
    words = phrase.split()
    normalized_words = [normalize_word(word) for word in words]
    return " ".join(normalized_words)


def lemmatize_phrase(phrase):
    return " ".join(morph_vocab.parse(word)[0].normal_form for word in phrase.split())


def find_lemmatized_phrases(sentence, phrase_lexicon):
    phrases_found = []
    fragments = split_sentence(sentence)

    for fragment in fragments:
        words = re.sub(r"[^\w\s]", "", fragment).lower().split()

        for i in range(len(words)):
            for j in range(i + 2, min(i + 6, len(words) + 1)):
                phrase = " ".join(words[i:j])
                lemma = normalize_phrase(lemmatize_phrase(phrase))

                if lemma in phrase_lexicon:
                    phrases_found.append((phrase, phrase_lexicon[lemma]))

    return phrases_found


def calculate_sentiment(sentence, lexicon, phrase_lexicon):
    fragments = split_sentence(sentence)

    sentiment_score = 0.0
    word_count = 0

    for fragment in fragments:
        phrases = find_lemmatized_phrases(fragment, phrase_lexicon)
        for phrase, score in phrases:
            sentiment_score += score
            word_count += 1

        for phrase, _ in phrases:
            fragment = re.sub(re.escape(phrase), "", fragment)

        words = re.findall(r"\b\w+\b", fragment.lower())

        negate = False
        for word in words:
            lemma = morph_vocab.parse(word)[0].normal_form

            if word in {"не", "ни", "никак", "нисколько"}:
                negate = True
                continue

            if lemma in lexicon:
                sentiment = lexicon[lemma]
                if negate and sentiment != 0:
                    sentiment = -sentiment
                sentiment_score += sentiment
                word_count += 1

            if negate and word not in {"не", "ни", "никак", "нисколько"}:
                negate = False

    return sentiment_score / word_count if word_count > 0 else 0


def preprocess_text(text):
    allowed_symbols = r"\-\—\«\»\"\{\}"
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        if line.strip().startswith("{") and line.strip().endswith("}"):
            processed_lines.append(line.strip())
            continue

        line = re.sub(r"\s+", " ", line.strip())
        line = re.sub(
            fr"[^а-яА-ЯёЁa-zA-Z0-9\s.,!?{allowed_symbols}]",
            "",
            line,
            flags=re.IGNORECASE,
        )
        processed_lines.append(line)

    return "\n".join(processed_lines)


def extract_sentences_and_sections(text):
    sentences = []
    sections = []
    sentence_counter = 1
    sentence_end = re.compile(r"([.!?])(\s+|$)")

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("{") and line.endswith("}"):
            section_name = line[1:-1].strip()
            sections.append((section_name, sentence_counter))
        else:
            parts = sentence_end.split(line)
            buffer = []

            for part in parts:
                buffer.append(part)
                if sentence_end.search(part):
                    sent = "".join(buffer).strip()
                    if sent:
                        sentences.append(sent)
                        sentence_counter += 1
                    buffer = []

            if buffer:
                sent = "".join(buffer).strip()
                if sent:
                    sentences.append(sent)
                    sentence_counter += 1

    return sentences, sections


def write_sections_file(sections, file_path="sections.txt"):
    with open(file_path, "w", encoding="utf-8") as file:
        for section, start in sections:
            file.write(f"'{section}' {start}\n")


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
    with open(filename, "w", encoding="utf-8") as file:
        for index, sentiment in enumerate(sentiments):
            file.write(f"{sentiment:+.2f} {sentences[index]}\n")

        if metrics:
            file.write("\n")
            file.write("Метрики графика:\n")
            for line in format_metric_lines(metrics):
                file.write(f"{line}\n")


def calculate_sentences_sentiments(sentences, section_positions, chunk_size, lexicon, phrase_lexicon):
    sentiments = []
    current_idx = 0
    section_ptr = 0
    sentence_count = len(sentences)
    section_count = len(section_positions)

    while current_idx < sentence_count:
        max_chunk_end = current_idx + chunk_size - 1
        chunk_end = min(max_chunk_end, sentence_count - 1)

        while section_ptr < section_count:
            section_start = section_positions[section_ptr] - 1

            if section_start > chunk_end:
                break

            if section_start >= current_idx:
                if section_start == current_idx:
                    section_ptr += 1
                    current_idx += 1
                    continue
                chunk_end = section_start - 1
                break
            section_ptr += 1

        chunk_end = max(current_idx, min(chunk_end, sentence_count - 1))

        chunk = sentences[current_idx : chunk_end + 1]
        if chunk:
            chunk_sentiments = [calculate_sentiment(sentence, lexicon, phrase_lexicon) for sentence in chunk]
            avg_sentiment = sum(chunk_sentiments) / len(chunk_sentiments)
            sentiments.extend([avg_sentiment] * len(chunk))

        current_idx = chunk_end + 1

    return sentiments


def roman_to_int(value):
    roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    prev_value = 0

    for char in reversed(value.upper()):
        current_value = roman_values.get(char)
        if current_value is None:
            return None
        if current_value < prev_value:
            total -= current_value
        else:
            total += current_value
            prev_value = current_value

    return total


def maybe_parse_section_number(label):
    cleaned = label.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return roman_to_int(cleaned)


def filter_section_ticks(section_positions, section_labels):
    if not SHOW_ONLY_EVEN_SECTION_LABELS:
        return section_positions, section_labels

    filtered_positions = []
    filtered_labels = []
    for position, label in zip(section_positions, section_labels):
        section_number = maybe_parse_section_number(label)
        if section_number is None or section_number % 2 == 0:
            filtered_positions.append(position)
            filtered_labels.append(label)

    return filtered_positions, filtered_labels


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
    ax.plot(x, sentiments, label="RuSentiLex", linewidth=RAW_LINE_WIDTH, color=RAW_LINE_COLOR)
    ax.plot(
        x,
        smoothed_uniform,
        label="Скользящее среднее",
        linewidth=SMOOTH_LINE_WIDTH,
        color=SMOOTH_LINE_COLOR,
    )

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

    tick_positions, tick_labels = filter_section_ticks(section_positions, section_labels)
    if section_positions:
        for position in section_positions:
            ax.axvline(
                x=position,
                color=SECTION_LINE_COLOR,
                linestyle=SECTION_LINE_STYLE,
                linewidth=SECTION_LINE_WIDTH,
                alpha=SECTION_LINE_ALPHA,
            )
        ax.set_xticks(tick_positions, tick_labels, rotation=45, fontsize=10)

    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="gray", alpha=0.7)

    title_suffix = title_note or CUSTOM_TITLE_NOTE or f"chunk_size = {chunk_size}"
    plot_title = CUSTOM_PLOT_TITLE or f"Кривые эмоциональной тональности ({title_suffix})"
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel(CUSTOM_X_LABEL or ("Главы" if section_positions else "Предложения"), fontsize=12, labelpad=12)
    ax.set_ylabel(CUSTOM_Y_LABEL, fontsize=12)
    ax.legend(fontsize=12)

    metrics_text = "    ".join(format_metric_lines({
        "volatility_amplitude": volatility_amplitude,
        "path_length": path_length,
        "mean_change": mean_change,
    }))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.28)
    fig.text(0.5, 0.06, metrics_text, ha="center", va="bottom", fontsize=10)

    fig.savefig(output_image, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    return {
        "volatility_amplitude": volatility_amplitude,
        "path_length": path_length,
        "mean_change": mean_change,
    }


def analyze_text_file(input_path, output_dir, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    processed_text = preprocess_text(text)
    sentences, sections = extract_sentences_and_sections(processed_text)
    if not sentences:
        raise RuntimeError("В выбранном файле не удалось выделить предложения для анализа.")

    write_sections_file(sections, output_dir / "sections.txt")
    section_labels = [label for label, _ in sections]
    section_positions = [position for _, position in sections]

    lexicon, phrase_lexicon = load_embedded_rusentilex()
    sentiments = calculate_sentences_sentiments(
        sentences,
        [],
        chunk_size,
        lexicon,
        phrase_lexicon,
    )

    results_path = output_dir / f"{input_path.name}_results.txt"
    plot_path = output_dir / f"{input_path.name}_sentiment_curve.png"
    metrics = plot_sentiment_curve(
        sentiments,
        sentences,
        section_positions,
        section_labels,
        chunk_size,
        plot_path,
    )
    save_results_to_text(sentences, sentiments, results_path, metrics)

    return {
        "result_txt": results_path,
        "result_png": plot_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone SentiSoft analysis script. "
            "Install matplotlib, natasha, numpy, scipy and pymorphy2 before running."
        )
    )
    parser.add_argument("input_path", help="Path to the input .txt file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Fragment size in sentences (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for generated TXT and PNG files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifacts = analyze_text_file(args.input_path, args.output_dir, args.chunk_size)
    print(f"TXT: {artifacts['result_txt']}")
    print(f"PNG: {artifacts['result_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _encode_lexicon_payload(lexicon_path: Path) -> str:
    return base64.b64encode(zlib.compress(lexicon_path.read_bytes(), level=9)).decode("ascii")


def build_export_script(*, default_chunk_size: int, lexicon_path: str | Path) -> str:
    lexicon_payload = _encode_lexicon_payload(Path(lexicon_path))
    return (
        EXPORT_SCRIPT_TEMPLATE.replace("__LEXICON_BASE64__", lexicon_payload)
        .replace("__DEFAULT_CHUNK_SIZE__", str(default_chunk_size))
    )


def export_analysis_script(
    script_path: str | Path,
    *,
    default_chunk_size: int,
    lexicon_path: str | Path,
) -> Path:
    target_path = Path(script_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        build_export_script(default_chunk_size=default_chunk_size, lexicon_path=lexicon_path),
        encoding="utf-8",
    )
    return target_path
