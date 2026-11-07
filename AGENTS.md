# Repository Guidelines

## Working agreements (Codex discipline)
- **Do not change behavior unless explicitly requested.** This project has a "behavior freeze" guarantee.
- Choose checks from **Testing Guidelines** below. Run `make verify` (fallback: `./tools/verify.sh`) for analysis/web/infrastructure changes and integration; use desktop checks for isolated desktop changes and document checks for prose-only edits.
- If you discover new setup steps, ports, env vars, commands, or verification steps while working —
  **update this AGENTS.md in the same diff** (keep it short and actionable).
- Keep diffs minimal. No “refactor while here”.
- If changes affect user-facing docs (setup/run steps, ports, UI flow, screenshots, new dependencies),
  update README.md in the same diff.

## Colab Sync Workflow
When updating the project from a new Google Colab Python export, use this workflow.

Recommended user request:
- `Я создал новый файл Colab: [filename].py. Согласно этому файлу обнови код в проекте.`
- Better explicit version: `Я создал новый файл Colab: [filename].py. Сначала отдели реальные изменения логики Colab от адаптаций проекта под web/Docker/export, затем перенеси только нужные изменения в runtime и export и запусти verify.`

Agent expectations:
- If the repo skill `sentilex-colab-sync` exists in `.agents/skills/sentilex-colab-sync/SKILL.md`, use it.
- Compare the new Colab file against both the previous Colab export and the current project code (`scripts/functions.py`, `app.py`, `scripts/export_script.py`).
- Treat notebook demo/smoke code after the usage boundary as non-spec by default unless the user explicitly says otherwise.
- Sync both runtime and exported standalone analysis when plotting/output formatting changes.
- After code changes, run `make verify`; if golden TXT changes intentionally, update the baseline in the same diff and state why.

## Behavior Freeze / Golden Regression Contract
We use a golden regression check to ensure sentiment logic + web workflow didn’t change.

Golden scenario:
1) Run app (Docker)
2) Upload `texts/chast__7.txt`
3) Download results TXT
4) Compare with golden baseline (default): `results/chast_7_result.txt`

Command:
- `./tools/golden_e2e.sh` (fails with diff if behavior changed)

If the golden output needs to change intentionally, document why and update the baseline file in the same PR.

## Emotional phase contract
- Keep original tone segmentation named “Эмоциональные фазы” and API `emotional_phases`. “Эмоциональные кластеры” now names the separate NRC marker feature (dominant-category chunks, not mathematical clustering).
- Preserve classification relative to `mean ± 0.5 * std`, minimum run length `max(2, chunk_size // 2)`, merge tie-breaking, inclusive sentence intervals, and length-weighted shares unless explicitly changing behavior.
- Phase shares differ from web sentiment distribution at fixed thresholds. Explain this distinction in user-facing docs; phase labels do not imply a sign relative to zero.
- Mirror phase detection, formatting, plotting and historical left-to-right chunk accumulation in runtime and standalone export. Run precision, golden and affected desktop checks; run export/reanalysis sequentially.
- Source of detail: [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md#эмоциональные-кластеры-фазы).

## Analysis Engine Change Log
When changing the analysis engine, update `docs/analysis_engine_changes.md` in the same diff.
This applies to changes in sentiment scoring, preprocessing, chunking, plotting metrics,
TXT result structure, export-script parity, or orchestration that affects analysis output.

Each entry must explain:
- which files/functions changed;
- how the behavior worked before;
- how it works after;
- why the change is needed;
- whether existing scoring/chunking/file formats changed;
- what validation was run.

## Project Structure & Module Organization
Core web application code lives in `app.py` (Flask routes, upload handling, orchestration). Text analysis logic and RuSentiLex integration live in `scripts/functions.py`, with the lexicon in `scripts/RuSentilex-2017.txt`. Desktop GUI code lives in `desktop/main.py`, with layout in `desktop/ui/main.ui` and styles in `desktop/style.qss`. HTML templates are in `templates/` (`upload.html`, `results.html`). Runtime input/output directories are `uploads/` and `results/`; treat files there as generated artifacts unless intentionally adding fixtures (the golden baseline is an exception).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `.venv/bin/pip install python-docx`: install the local dependency required to rebuild `article/software_section/Sentilex_software_section.docx` with `python article/software_section/build_docx.py`.
- `python app.py`: run the web app locally on `http://127.0.0.1:5000` (debug mode).
- `docker build -t sentilex .`: build a container image from `Dockerfile`.
- `docker run -p 5000:5000 sentilex`: run the app in Docker.
- `make verify` (or `./tools/verify.sh`): run automated checks (golden regression, etc).
- `pip install -r desktop/requirements.txt`: install desktop-only dependencies (PySide6) without changing web/Docker deps.
- `.venv/bin/python desktop/main.py`: run desktop MVP locally.
- `.venv/bin/python desktop/smoke_reanalysis.py`: run non-GUI desktop smoke for repeated analysis refresh (TXT/PNG/PY updated on second run).
- `.venv/bin/python desktop/smoke_export_script.py`: verify that exported standalone Python code reproduces the desktop TXT result outside the app.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_layout.py`: verify adaptive desktop layout (stacked/wide) at representative window sizes without a display.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_chunk_slider.py`: verify that desktop chunk-size slider options are derived from the selected file's sentence count.
- `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/render_main.py`: render `desktop/ui/main.ui` to `docs/design/desktop_parity.png` without a display (WSL/headless).
- `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/render_variants.py`: render desktop GUI design variants to `docs/design/*.png` without a display (WSL/headless).
- `powershell -ExecutionPolicy Bypass -File tools/build_windows.ps1`: build Windows desktop `onedir` bundle (`dist/SentiSoft/`) via PyInstaller (use Python 3.10/3.11 for pymorphy2 compatibility).
- `powershell -ExecutionPolicy Bypass -File tools/build_windows.ps1 -OneFile`: build Windows desktop `onefile` executable (`dist/SentiSoft.exe`) (use Python 3.10/3.11 for pymorphy2 compatibility).

## Coding Style & Naming Conventions
Use Python 3.10+ and follow PEP 8 style (4-space indentation, `snake_case` for functions/variables, `UPPER_CASE` for constants like `UPLOAD_FOLDER`). Keep Flask route handlers in `app.py` thin, and place processing logic in `scripts/functions.py`. Prefer small, single-purpose functions and explicit file encodings (`encoding='utf-8'`) for text I/O.

## Testing Guidelines

- `make verify` also checks historical TXT rounding in Docker; if `.venv/bin/python` exists, it checks host rounding too. Run `.venv/bin/python tools/smoke_chunk_precision.py` for the focused runtime/export precision regression.
- If host loopback cannot reach the Docker port, use `GOLDEN_HTTP_MODE=container make verify` (real HTTP upload/download inside the container; published-port reachability remains unverified). For a missing Buildx plugin, `DOCKER_BUILDKIT=0` selects the legacy builder.
- Docker build context includes only web runtime sources (`.dockerignore`); provide test inputs from the host through the golden upload flow.
- Run `desktop/smoke_reanalysis.py` and `desktop/smoke_export_script.py` sequentially: they write the same result paths.

Choose checks for the actual change; do not rerun a passing full suite without a new change or unresolved concern.

| Change | Required validation |
|---|---|
| Prose, AGENTS, skills only | Check changed facts, links, YAML where applicable, and command paths; no application suite. |
| Desktop `.ui`/QSS/layout | Render and inspect the affected main UI or variants; run `desktop/smoke_layout.py` offscreen for layout changes. |
| Desktop chunk controls | Run `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_chunk_slider.py`; inspect affected UI. Preserve file-dependent choices/defaults from `desktop/main.py`, not a fixed web range. |
| Desktop reanalysis/result refresh | Run `.venv/bin/python desktop/smoke_reanalysis.py`; observe the affected flow. |
| Desktop standalone export | Run `.venv/bin/python desktop/smoke_export_script.py`. |
| Analysis, preprocessing, chunking algorithms, shared exports, Flask routes/templates, dependencies, Docker/verification infrastructure | Run `make verify` (or `./tools/verify.sh`), which includes the golden regression. Add the relevant desktop check if affected. |
| Preparing code changes for integration | Run the full verification once after the last relevant change, plus affected desktop checks. |

For other isolated desktop behavior, use the closest smoke/targeted check and exercise the affected interaction; a web golden alone does not validate desktop behavior. If a command cannot run, report the actual blocker and which contract remains unverified.

For visible web changes, observe the affected browser flow and inspect before/after output. For broader workflow checks, upload short + long `.txt` and verify:
- upload validation (`.txt` only),
- result text generation,
- sentiment plot generation,
- route behavior for `/`, `/results/<filename>`, and `/download/<filename>`.

## Commit & Pull Request Guidelines
The current history uses short, imperative commit messages (e.g., `Update README.md`, `Create LICENSE`). Keep that style, and prefer focused commits per concern. For PRs, include:
- clear summary of behavioral changes (or state "no behavior change"),
- verification steps/commands (include `make verify` output when required by the matrix),
- before/after screenshots for template/UI changes,
- linked issue or rationale for non-trivial changes.

## Execution plans

For feature work, multi-step refactors, or any task that mentions `PLANS.md`:
- read `/PLANS.md` before touching code;
- treat `/PLANS.md` as the implementation source of truth;
- keep `Progress`, `Surprises & Discoveries`, `Decision Log`, and `Outcomes & Retrospective` current;
- if the real repository differs from the plan, update `/PLANS.md` first, then continue implementation;
- run the validation commands specified in `/PLANS.md` after each milestone.

## NRC desktop emotions
- External NRC ZIP/XLSX must never be committed, bundled or embedded into exported PY. Ignore NRC archives and Windows build artifacts.
- Run `.venv/bin/python desktop/smoke_emotions.py` for marker counts, ties, shares and external archive import. Desktop requires openpyxl from desktop/requirements.txt.

- Windows builds from WSL UNC sources use `%LOCALAPPDATA%/SentilexBuild/venv`; `-VenvPath` accepts an absolute Windows path. The builder bootstraps pip if needed.
- Frozen verification: set `SENTILEX_SMOKE_INPUT` and `SENTILEX_SMOKE_REPORT` to absolute paths; optional `SENTILEX_SMOKE_NRC` selects an external archive. The exe runs the actual worker, renders its results, writes a JSON manifest and PNG next to the report, and exits.
- GUI regression: `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_emotion_gui.py` (requires the local external archive).

- `powershell -ExecutionPolicy Bypass -File tools/smoke_frozen.ps1 [-NrcPath <ZIP-or-XLSX>]` tests a local NTFS copy of dist/SentiSoft.exe; manifest/screenshot are copied to dist. It compares tone TXT after CRLF normalization.

- Desktop numeric chunk input accepts 1…sentence_count and inserts exact values into the file-derived slider choices. Initial choices include 1..10 (within text length), then tens and the final sentence count; preserve existing defaults. Verify two-way synchronization with smoke_chunk_slider.py. Chapter-label stride is ordinal (1, 1+N, …), defaults to 1, preserves chapter boundaries and TXT; export mirrors it through CHAPTER_LABEL_STEP.

- `.venv/bin/python desktop/smoke_plot_options.py` validates dense chapter labels, full plot legend and exported default chapter stride without changing metrics.

- NRC fill rendering: `.venv/bin/python desktop/smoke_emotion_fills.py` checks curve clipping, solid palette, vertical ties and chunk boundaries.

- NRC reports: `count_chunks` retains sentence/marker offsets and sums sentence counts into chunk counts. Keep longest-match/no-overlap rules and independent sentence/chunk leaders. HTML is escaped, offline and displayed in the “Маркеры кластеров” tab; clear it on rerun, error and tone-only analysis. Frozen smoke checks HTML and captures an additional `_markers.png`.

- A running Windows exe cannot be overwritten. Do not stop the user application: stage the PyInstaller build with `--distpath` on local NTFS and copy it to `dist/SentiSoft-clusters.exe`. `tools/smoke_frozen.ps1 -ExePath <absolute Windows path>` verifies an alternate executable; default remains dist/SentiSoft.exe.
