import os
import sys
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
import export_script
import functions

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

ALLOWED_EXTENSIONS = {"txt"}
LEXICON_PATH = "scripts/RuSentilex-2017.txt"

# Глобальные переменные RuSentilex
lexicon = None
phrase_lexicon = None

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_rusentilex():
    global lexicon, phrase_lexicon
    file_path = LEXICON_PATH
    
    if os.path.exists(file_path):
        lexicon, phrase_lexicon = functions.load_rusentilex(file_path)
        print("✅ RuSentilex загружен успешно!")
    else:
        raise FileNotFoundError(f"❌ Файл {file_path} не найден.")

initialize_rusentilex()

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        chunk_size = request.form.get("chunk_size", 50, type=int)

        if not file or file.filename == "":
            return "Файл не выбран!"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            process_text(file_path, filename, chunk_size)

            return redirect(url_for("results", filename=filename))
    
    return render_template("upload.html")

def process_text(file_path, filename, chunk_size, export_script_file=False):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    processed_text = functions.preprocess_text(text)
    sentences, sections = functions.extract_sentences_and_sections(processed_text)
    functions.write_sections_file(sections)
    section_labels = [label for label, _ in sections]
    section_positions = [position for _, position in sections]

    sentiments = functions.calculate_sentences_sentiments(
        sentences,
        [],
        chunk_size,
        lexicon,
        phrase_lexicon,
    )

    results_path = os.path.join(app.config["RESULT_FOLDER"], f"{filename}_results.txt")
    plot_path = os.path.join(app.config["RESULT_FOLDER"], f"{filename}_sentiment_curve.png")

    metrics = functions.plot_sentiment_curve(
        sentiments,
        sentences,
        section_positions,
        section_labels,
        chunk_size,
        plot_path,
    )

    functions.save_results_to_text(sentences, sentiments, results_path, metrics)
    artifacts = {
        "result_txt": results_path,
        "result_png": plot_path,
    }

    if export_script_file:
        script_path = os.path.join(app.config["RESULT_FOLDER"], f"{filename}_analysis_code.py")
        export_script.export_analysis_script(
            script_path,
            default_chunk_size=chunk_size,
            lexicon_path=LEXICON_PATH,
        )
        artifacts["result_py"] = script_path

    return artifacts
    
@app.route("/results/<filename>")
def results(filename):
    return render_template("results.html", filename=filename)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(app.config["RESULT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    
