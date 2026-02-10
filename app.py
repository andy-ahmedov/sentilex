import os
import sys
import time

from flask import flash, redirect, render_template, request, send_from_directory, url_for, Flask
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
import functions

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sentilex-dev-key")

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {"txt"}
DEFAULT_CHUNK_SIZE = 50
MIN_CHUNK_SIZE = 10
MAX_CHUNK_SIZE = 500

# Глобальные переменные RuSentilex
lexicon = None
phrase_lexicon = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def initialize_rusentilex():
    global lexicon, phrase_lexicon
    file_path = "scripts/RuSentilex-2017.txt"

    if os.path.exists(file_path):
        lexicon, phrase_lexicon = functions.load_rusentilex(file_path)
        print("RuSentilex loaded successfully")
    else:
        raise FileNotFoundError(f"File {file_path} was not found.")


initialize_rusentilex()


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        chunk_size_raw = (request.form.get("chunk_size", str(DEFAULT_CHUNK_SIZE)) or "").strip()

        try:
            chunk_size = int(chunk_size_raw)
        except ValueError:
            flash("Размер фрагмента должен быть целым числом.", "danger")
            return redirect(url_for("upload_file"))

        if chunk_size < MIN_CHUNK_SIZE or chunk_size > MAX_CHUNK_SIZE:
            flash(
                f"Размер фрагмента должен быть в диапазоне {MIN_CHUNK_SIZE}-{MAX_CHUNK_SIZE}.",
                "danger",
            )
            return redirect(url_for("upload_file"))

        if not file or file.filename == "":
            flash("Выберите .txt файл для анализа.", "danger")
            return redirect(url_for("upload_file"))

        if not allowed_file(file.filename):
            flash("Поддерживаются только файлы формата .txt.", "danger")
            return redirect(url_for("upload_file"))

        filename = secure_filename(file.filename)
        if not filename:
            flash("Некорректное имя файла.", "danger")
            return redirect(url_for("upload_file"))

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        try:
            file.save(file_path)
            started_at = time.perf_counter()
            metrics = process_text(file_path, filename, chunk_size)
            elapsed = time.perf_counter() - started_at
        except Exception:
            app.logger.exception("Failed to process file: %s", filename)
            flash("Не удалось выполнить анализ. Попробуйте другой файл.", "danger")
            return redirect(url_for("upload_file"))

        return redirect(
            url_for(
                "results",
                filename=filename,
                chunk_size=chunk_size,
                sentence_count=metrics["sentence_count"],
                processing_time=f"{elapsed:.2f}",
            )
        )

    return render_template("upload.html", default_chunk_size=DEFAULT_CHUNK_SIZE)


def process_text(file_path, filename, chunk_size):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    processed_text = functions.preprocess_text(text)
    sentences = functions.split_into_sentences(processed_text)

    # Загружаем разделы из файла sections.txt
    section_labels, section_positions = [], []
    with open("sections.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                label, position = line.split(" ", 1)
                label = label.strip("'")
                section_labels.append(label)
                section_positions.append(int(position))

    sentiments1 = functions.calculate_sentences_sentiments(
        sentences,
        section_positions,
        chunk_size,
        lexicon,
        phrase_lexicon,
    )
    sentiments2 = functions.calculate_sentences_sentiments(
        sentences,
        section_positions,
        2 * chunk_size,
        lexicon,
        phrase_lexicon,
    )

    min_len = min(len(sentiments1), len(sentiments2), len(sentences))
    sentiments = [(a + b) / 2 for a, b in zip(sentiments1[:min_len], sentiments2[:min_len])]
    sentences = sentences[:min_len]

    results_filename = f"{filename}_results.txt"
    plot_filename = f"{filename}_sentiment_curve.png"
    results_path = os.path.join(app.config["RESULT_FOLDER"], results_filename)
    plot_path = os.path.join(app.config["RESULT_FOLDER"], plot_filename)

    functions.save_results_to_text(sentences, sentiments, results_path)
    functions.plot_sentiment_curve(
        sentiments,
        sentences,
        section_positions,
        section_labels,
        2 * chunk_size,
        plot_path,
    )

    return {
        "sentence_count": len(sentences),
        "results_filename": results_filename,
        "plot_filename": plot_filename,
    }


@app.route("/results/<filename>")
def results(filename):
    chunk_size = request.args.get("chunk_size", type=int)
    sentence_count = request.args.get("sentence_count", type=int)
    processing_time = request.args.get("processing_time", type=float)
    processing_time_display = f"{processing_time:.2f} с" if processing_time is not None else "—"

    return render_template(
        "results.html",
        filename=filename,
        chunk_size=chunk_size if chunk_size is not None else "—",
        sentence_count=sentence_count if sentence_count is not None else "—",
        processing_time=processing_time_display,
        result_filename=f"{filename}_results.txt",
        plot_filename=f"{filename}_sentiment_curve.png",
    )


@app.route("/result-file/<path:filename>")
def view_result_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=False)


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
