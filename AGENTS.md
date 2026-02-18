# Repository Guidelines

## Project Structure & Module Organization
The codebase is a small Flask application centered on `app.py`.

- `app.py`: web entrypoint, routes, upload/result flow, and runtime setup for `uploads/` and `results/`.
- `scripts/functions.py`: sentiment-processing pipeline (tokenization, lemmatization, scoring, plotting).
- `scripts/RuSentilex-2017.txt`: sentiment lexicon used at startup.
- `templates/upload.html`, `templates/results.html`: UI templates.
- `Dockerfile`, `requirements.txt`: container and dependency definitions.

Keep new Python modules under `scripts/` unless they are web-layer concerns that belong in `app.py`.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate local virtualenv.
- `pip install -r requirements.txt`: install dependencies.
- `python app.py`: run locally at `http://127.0.0.1:5000`.
- `docker build -t sentilex .`: build container image.
- `docker run -p 5000:5000 sentilex`: run app in Docker.
- `python -m compileall app.py scripts`: quick syntax validation before commit.

## Coding Style & Naming Conventions
Use PEP 8 as the baseline:

- 4-space indentation, snake_case for functions/variables, UPPER_CASE for module-level constants.
- Keep route handlers in `app.py` focused on HTTP flow; move text-analysis logic into `scripts/functions.py`.
- Prefer small, pure helper functions for NLP transformations.
- Use UTF-8 file encoding and explicit `encoding="utf-8"` for file I/O.

## Testing Guidelines
There is currently no automated test suite in this repository. Until one is added:

- Run `python -m compileall app.py scripts`.
- Manually test upload, analysis, and download flows in browser.
- Validate both local and Docker execution paths for behavior parity.

If tests are introduced, place them in `tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
Git history uses short, imperative commit subjects (for example, `Update README.md`, `Create LICENSE`).

- Keep commit titles concise and action-oriented.
- Group related changes into a single commit.
- PRs should include: purpose, key behavior changes, manual test steps, and UI screenshots when templates are changed.
- Link related issues/tasks when available.
