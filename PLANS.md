# Adaptive emotional phase segmentation for Sentilex

This PLANS.md is the execution source of truth for the emotional-cluster upgrade in Sentilex.
It is a living document and must be updated as work proceeds.

Codex operating rules for this plan:
- Read this file fully before editing code.
- Read AGENTS.md, PROJECT_OVERVIEW.md, README.md, app.py, scripts/functions.py, and scripts/export_script.py before changing implementation.
- Keep diffs tightly scoped to emotional phase segmentation.
- Do not change the existing sentence-level sentiment model, chunk aggregation logic, or file formats unless the plan explicitly requires it.
- If repository reality differs from this plan, pause implementation, update this PLANS.md first, record the discrepancy in `Surprises & Discoveries` and `Decision Log`, then continue.
- When changing analysis-engine behavior, update `docs/analysis_engine_changes.md` according to AGENTS.md.
- After each milestone, run the validation commands listed in this file, fix failures immediately, and update `Progress`.
- Do not leave partially true status notes. `Progress` must always reflect the actual current state.

## Purpose

Sentilex already produces a sentence/chunk sentiment curve and derived plot metrics.
This change upgrades the project from “curve visualization” to “adaptive emotional phase segmentation”.

After this change:
- the smoothed emotional curve is segmented into stable positive, neutral, and negative phases;
- these phases are highlighted directly on the PNG plot;
- the TXT result includes a new phase-summary block;
- the exported standalone analysis script reproduces the same behavior.

The goal is to create a conference-ready feature that looks methodologically stronger than fixed threshold coloring while remaining lightweight, dependency-free, and consistent with the current architecture.

## Progress

- [x] Pre-push repair: preserve historical chunk accumulation across Python versions, rebuild Docker without unused compiler packages, expose verification files to Git, and rerun golden plus desktop checks.

- [x] Initial repository audit completed and this plan reconciled against the real codebase.
- [x] Phase detection helper API added to `scripts/functions.py`.
- [x] Plot highlighting added to `scripts/functions.py::plot_sentiment_curve`.
- [x] Phase metrics added to TXT output in `scripts/functions.py::save_results_to_text`.
- [x] Export parity implemented in `scripts/export_script.py`.
- [x] Golden/regression impact evaluated and baseline updated if required.
- [x] Analysis-engine changes documented in `docs/analysis_engine_changes.md`.
- [x] Validation commands executed successfully.
- [x] README / PROJECT_OVERVIEW / AGENTS notes updated if behavior or operator workflow changed.
- [x] Final retrospective recorded.

## Surprises & Discoveries

- 2026-11-07: Host requests to the published Docker port failed (proxy HTTP 502 / direct connection reset), while HTTP inside the container returned 200. Added explicit `GOLDEN_HTTP_MODE=container` for the same upload/download golden comparison; host mode remains the default. This does not verify published-port reachability.

- 2026-11-07: Pre-push verification found Debian security-package 404 errors in the Docker compiler-install layer, a 2.37 GB unfiltered build context, and Python 3.12 `sum()` rounding differences. The existing Python 3.10 container reproduces the current golden exactly; sequential addition also restores exact golden output on Python 3.12.

- Observation: `scripts/export_script.py` mirrors the main analysis pipeline but also contains export-specific plot customization hooks.
  Evidence: `RAW_LINE_COLOR`, `SMOOTH_LINE_COLOR`, `SHOW_ONLY_EVEN_SECTION_LABELS`, `CUSTOM_*`, and `filter_section_ticks(...)` are present in the export template and should be preserved when phase highlighting is added.
- Observation: `AGENTS.md`, `PLANS.md`, and `.gitignore` were ignored by `.gitignore`.
  Evidence: `git check-ignore -v AGENTS.md PLANS.md .gitignore` reported matching `.gitignore` entries; execution-plan updates would not be visible in normal git status until these ignore rules are removed.
- Observation: on a large real-text desktop plot, the first phase-background defaults were too subtle.
  Evidence: user screenshot with `chunk_size = 500` showed phase colors difficult to distinguish from the mean corridor and dense section markers.

## Decision Log

- Decision (2026-11-07): preserve historical left-to-right chunk accumulation explicitly in runtime and export. Keep the golden unchanged. Remove unused apt build tools from the web image and restrict its context to runtime files; keep the existing Python 3.10 base. Add a rounding regression check to verification.

- Decision: Track `AGENTS.md` and `PLANS.md` as normal repository files.
  Rationale: The execution-plan workflow requires `PLANS.md` progress, discoveries, decisions, and retrospectives to be updated in the same diff as feature work. Keeping those files ignored would make that workflow unreliable.
  Date/Author: 2026-06-16 / Codex
- Decision: Preserve exported-script customization hooks while adding emotional phase rendering.
  Rationale: The export script is behaviorally aligned with the main engine but has user-editable plotting constants; phase work should extend that export surface without removing existing customization.
  Date/Author: 2026-06-16 / Codex
- Decision: Detect phases with an adaptive neutral corridor around the smoothed curve mean.
  Rationale: The detector uses `mean(smoothed_curve) +/- 0.5 * std(smoothed_curve)` instead of fixed sentiment thresholds, then merges runs shorter than `max(2, chunk_size // 2)` into neighboring stable phases. This keeps phase boundaries tied to the current text and current smoothing granularity.
  Date/Author: 2026-06-16 / Codex
- Decision: Append TXT phase data as a separate block below existing graph metrics.
  Rationale: This keeps sentence-by-sentence output and the existing `Метрики графика` block stable while making phase counts, phase shares, and sentence-coordinate intervals available for reporting.
  Date/Author: 2026-06-16 / Codex
- Decision: Mirror phase logic directly inside the standalone export template.
  Rationale: The exported `.py` file must run without importing the project package, so it needs its own copy of the phase helpers, TXT formatters, and plot background rendering while preserving existing editable plot constants.
  Date/Author: 2026-06-16 / Codex
- Decision: Keep the phase algorithm unchanged, but strengthen PNG phase presentation.
  Rationale: User testing showed the phase regions were technically present but not readable enough on dense literary plots. The fix should improve visual hierarchy without changing sentence scoring, chunk aggregation, or phase boundaries.
  Date/Author: 2026-06-16 / Codex

## Outcomes & Retrospective

- Documentation follow-up (2026-11-07): README now explains emotional clusters as phases, automatic analysis workflow, relative labels, interval coordinates, length-weighted shares, rounding and chunk-size sensitivity. PROJECT_OVERVIEW records detector and merge contracts; AGENTS preserves runtime/export parity and correct Windows build paths. Russian change log now includes the precision repair and actual verification limits. Documentation checked against source, local links and command paths; no application rerun for prose-only changes.

- Pre-push repair (2026-11-07): the precision regression failed before the fix and passed after on host Python 3.12 and container Python 3.10. Local full TXT matches the existing golden byte for byte. Export and reanalysis smoke checks passed sequentially. `GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify` passed, including Docker rebuild, real HTTP upload/download, and exact golden comparison. The runtime build context dropped from 2.37 GB to 1.44 MB; dependencies installed without apt/compiler packages. No additional baseline changes were made. Host published-port reachability remains an environment limitation; the explicit container mode is documented. A user-local Buildx link now points to the installed distro binary instead of the stale Docker Desktop link.

- Milestone A: repository audit completed; no analysis implementation code changed.
- Milestone A validation: `make verify` passed on 2026-06-16. The run completed Python compileall and golden e2e regression; downloaded TXT matched `results/chast_7_result.txt`.
- Milestone B: adaptive phase helper API added to `scripts/functions.py`. The helpers classify smoothed points, merge short phase runs, build phase span dictionaries, and calculate aggregate phase metrics.
- Milestone B behavior impact: sentence-level sentiment scoring and chunk aggregation are unchanged. TXT and PNG visible output are unchanged at this stage; `plot_sentiment_curve(...)` now returns additional phase data in its metrics dictionary for later milestones.
- Milestone B validation: `python -m py_compile scripts/functions.py` passed; a direct helper smoke check produced ordered negative/neutral/positive phases; `make verify` passed on 2026-06-16 and golden TXT still matched `results/chast_7_result.txt`, so no baseline update was needed.
- Milestone C: `scripts/functions.py::plot_sentiment_curve(...)` now renders emotional phases as subtle background intervals behind the existing raw and smoothed curves.
- Milestone C behavior impact: PNG visual output changed intentionally. Sentence-level sentiment scoring, chunk aggregation, TXT structure, and existing graph metric calculations are unchanged.
- Milestone C validation: `python -m py_compile scripts/functions.py` passed; a direct PNG smoke check wrote `/tmp/sentilex_phase_smoke.png` with visible phase backgrounds while preserving the existing curves and metrics block; `make verify` passed on 2026-06-16 and golden TXT still matched `results/chast_7_result.txt`, so no baseline update was needed.
- Milestone D: `scripts/functions.py::save_results_to_text(...)` now appends an `Эмоциональные фазы` block below the existing graph metrics when phase metrics are present.
- Milestone D behavior impact: TXT output changed intentionally by adding aggregate phase metrics and compact sentence-coordinate phase intervals. Sentence-level sentiment scoring, chunk aggregation, PNG rendering, and existing graph metrics are unchanged.
- Milestone D validation: `python -m py_compile scripts/functions.py` passed; a direct TXT smoke check wrote `/tmp/sentilex_phase_txt_smoke.txt` with the new phase block; the first `make verify` run failed only because golden TXT lacked the intentional appended phase block; `results/chast_7_result.txt` was updated from the golden e2e downloaded TXT; the second `make verify` passed on 2026-06-16.
- Milestone E: `scripts/export_script.py` now generates standalone scripts with the same adaptive phase helpers, PNG phase backgrounds, and TXT phase summary blocks as the main engine.
- Milestone E behavior impact: exported standalone script output changed intentionally to match the phase-aware main engine. Main web/desktop scoring, chunk aggregation, and golden TXT output are unchanged in this milestone.
- Milestone E validation: `python -m py_compile scripts/export_script.py` passed; generated `/tmp/sentilex_export_phase_smoke.py` compiled; local parity check on `texts/chast__7.txt` with `chunk_size=50` matched main-engine TXT exactly in the same Python environment; `.venv/bin/python desktop/smoke_export_script.py` passed; `make verify` passed on 2026-06-16 with no baseline update.
- Milestone F: final validation completed. No additional code or baseline changes were needed in this milestone.
- Milestone F validation: `make verify` passed on 2026-06-16; `.venv/bin/python desktop/smoke_export_script.py` passed and confirmed exported standalone TXT parity; `.venv/bin/python desktop/smoke_reanalysis.py` passed and confirmed repeated desktop analysis refreshes TXT, PNG, and PY outputs.
- Final retrospective: emotional phase segmentation is implemented end-to-end. The main engine now derives positive, neutral, and negative phases from the existing smoothed sentiment curve using an adaptive `mean +/- 0.5 * std` neutral corridor and merges spans shorter than `max(2, chunk_size // 2)`. PNG output now highlights phase regions, TXT output now appends aggregate phase metrics and sentence-coordinate intervals, and exported standalone scripts reproduce the same phase-aware behavior. Existing sentence scoring and chunk aggregation were not changed. The golden TXT baseline changed intentionally in Milestone D only because the new `Эмоциональные фазы` block became part of the expected output. `docs/analysis_engine_changes.md` contains before/after entries for the analysis-engine changes.
- Follow-up visual refinement: PNG phase backgrounds were made more readable, section markers and the mean corridor were visually de-emphasized, and a compact phase summary was added below the existing graph metrics. The same presentation defaults were mirrored into the standalone export template. The phase detection algorithm, sentence scoring, chunk aggregation, and TXT output structure remain unchanged.

## Context and orientation

Current known architecture:
- Web and desktop share the same analysis engine in `scripts/functions.py`.
- `app.py::process_text(...)` orchestrates preprocessing, sentence extraction, sentiment computation, PNG generation, TXT saving, and optional export script generation.
- `scripts/functions.py::plot_sentiment_curve(...)` already computes:
  - `smoothed_uniform` via `uniform_filter1d`
  - `volatility_amplitude`
  - `path_length`
  - `mean_change`
  - `mean_value`
  - `upper_bound` / `lower_bound` for `mean ± volatility`
- `scripts/export_script.py` contains a mirrored copy of analysis and plotting logic and must stay behaviorally aligned.
- The current result page depends mainly on filename/chunk metadata, sentiment distribution, and the generated PNG/TXT. UI exposure of new phase metrics is optional and out of scope unless implementation is trivial and low risk.
- The project has a regression contract (`make verify` / `./tools/verify.sh`) and a behavior-freeze expectation. If the TXT golden output changes as an intended result of this feature, update the baseline in the same change and explain why.

Primary files in scope:
- `scripts/functions.py`
- `scripts/export_script.py`
- optional: `app.py` only if required for passing through new metrics
- optional: documentation files if behavior description changes

Files presumed out of scope unless required by validation or parity:
- `desktop/ui/*`
- `desktop/style.qss`
- web templates

## Scope boundaries

In scope:
- Adaptive emotional phase segmentation from the existing smoothed sentiment curve
- Plot highlighting for phases
- Phase metrics in TXT
- Exported-script parity
- Regression/doc updates required by the feature

Out of scope:
- New NLP models
- NER / graph extraction / semantic graphs
- New external dependencies
- Full change-point-detection frameworks
- Major UI redesign
- Changes to core lexical sentiment scoring

## Implementation strategy

Implement a lightweight adaptive phase detector on top of the existing smoothed curve.

Proposed method:
1. Reuse the existing `smoothed_uniform`.
2. Compute `mean_value` and `volatility_amplitude` from the smoothed curve.
3. Define an adaptive neutral corridor around the mean.
4. Label candidate states as positive / neutral / negative based on excursions relative to that corridor.
5. Enforce persistence:
   - a phase must survive for a minimum span before it is accepted;
   - very short spikes are merged into neighboring stable phases.
6. Return phase spans and summary metrics.
7. Render phase spans on the plot with subtle background highlighting.
8. Append a dedicated phase block to the TXT output.
9. Port the same logic into the export script template.

Recommended phase detector contract:
- Input:
  - `smoothed_curve`
  - `chunk_size`
  - optional sensitivity parameters with safe defaults
- Output:
  - ordered list of phase spans, each span containing:
    - `start_index`
    - `end_index`
    - `phase_type` (`positive`, `neutral`, `negative`)
    - optional summary statistics for the span

Recommended defaults:
- Avoid hardcoded global thresholds like `0.15` / `-0.15`.
- Use an adaptive corridor derived from the current text’s smoothed-curve distribution.
- Tie minimum stable phase length to smoothing/chunk settings so the detector scales with the existing analysis granularity.
- Prefer simple deterministic rules over opaque heuristics.

## Milestones

### Milestone A

Audit the actual repository state before touching behavior.

Tasks:
- Confirm the live signatures and data flow in:
  - `app.py::process_text`
  - `scripts/functions.py::plot_sentiment_curve`
  - `scripts/functions.py::save_results_to_text`
  - `scripts/export_script.py` plot/template code
- Confirm whether any local AGENTS instructions or hidden repo rules change the required workflow.
- If the repository differs from assumptions in this plan, update:
  - `Context and orientation`
  - `Scope boundaries`
  - `Progress`
  - `Decision Log`
  before implementation starts.

Acceptance:
- This plan matches the actual codebase closely enough to guide implementation without guesswork.

### Milestone B

Add adaptive phase segmentation to the main engine.

Tasks:
- In `scripts/functions.py`, add a small, testable helper group, for example:
  - `classify_phase_points(...)`
  - `merge_short_phase_runs(...)`
  - `build_emotional_phases(...)`
  - `calculate_phase_metrics(...)`
- Keep these helpers pure where possible.
- Base all segmentation on the existing smoothed curve.
- Do not change the existing raw sentiment list generation.
- Return phase metrics in a backward-compatible dictionary extension.

Recommended metrics:
- `phase_count_total`
- `phase_count_positive`
- `phase_count_negative`
- `phase_count_neutral`
- `phase_avg_len`
- `phase_max_positive_len`
- `phase_max_negative_len`
- `phase_positive_share`
- `phase_negative_share`
- `phase_neutral_share`

Acceptance:
- A representative text produces a stable ordered list of emotional phases.
- Tiny one-off spikes do not produce visually noisy micro-phases.
- Existing volatility/path-length/mean-change metrics remain available.

### Milestone C

Render phases on the PNG plot.

Tasks:
- Extend `plot_sentiment_curve(...)` to draw subtle phase backgrounds using span-based rendering.
- Ensure the existing curve lines, mean band, extrema annotations, and section markers remain readable.
- Keep colors conservative and presentation-oriented rather than flashy.
- Update the returned metrics dictionary to include the new phase metrics.

Acceptance:
- The PNG clearly shows stable emotional regions without obscuring the original curve.
- Section labels, extrema annotations, and the lower metrics block remain legible.

### Milestone D

Extend TXT output with a phase summary.

Tasks:
- Keep the existing sentence-by-sentence output unchanged.
- Keep the existing “Метрики графика” block.
- Add a new dedicated block below it, for example:
  - “Эмоциональные фазы”
  - aggregate phase metrics
  - optional compact list of spans in sentence coordinates
- If a stronger minimal-diff strategy is needed, prefer aggregate metrics first and span listings second.

Acceptance:
- TXT remains human-readable.
- The added block is useful for conference reporting and reproducible comparison across texts.

### Milestone E

Restore export parity.

Tasks:
- Mirror the new helper logic and metric names into `scripts/export_script.py`.
- Ensure the exported standalone script generates phase-aware PNG/TXT artifacts compatible with the main application behavior.
- Keep customization hooks intact where possible.

Acceptance:
- Exported script behavior matches the main engine for the same input and chunk size.

### Milestone F

Validation, regression handling, and documentation.

Tasks:
- Run project validation commands.
- If the golden TXT necessarily changes, update the golden artifact in the same change and record why.
- Run desktop smoke checks relevant to reanalysis/export parity if the repository contains them and they are part of the documented workflow.
- Update docs if operator-facing behavior changed.

Acceptance:
- Validation passes, or any intentional baseline changes are explicitly justified and committed together.

## Validation and acceptance

Run from repository root unless the real repo layout requires otherwise.

Primary validation:
- `make verify`

Fallback if the primary command is unavailable:
- `./tools/verify.sh`

Recommended additional checks if relevant commands/files exist:
- `.venv/bin/python desktop/smoke_reanalysis.py`
- `.venv/bin/python desktop/smoke_export_script.py`
- `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_layout.py`
- `QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_chunk_slider.py`

Manual functional check:
1. Run analysis on one known literary `.txt`.
2. Confirm the PNG now shows phase backgrounds.
3. Confirm the TXT contains the new phase summary block.
4. Re-run with a different `chunk_size` and verify that phases remain plausible and stable rather than becoming random noise.
5. If export is enabled, generate the standalone script and verify it reproduces the updated TXT/PNG behavior.

Expected evidence to record in this plan:
- commands executed
- pass/fail result
- if baseline changed, a short explanation of why
- any surprising sensitivity to `chunk_size`
- confirmation that `docs/analysis_engine_changes.md` explains engine changes in before/after terms

## Working loop for Codex

At the start of each session:
1. Read `AGENTS.md`.
2. Read this `PLANS.md`.
3. Read the implementation files named in `Context and orientation`.
4. Reconcile assumptions vs reality.
5. Update this file if assumptions are no longer true.
6. Only then edit code.

After completing any meaningful sub-step:
1. Update `Progress`.
2. Record notable surprises.
3. Record design choices in `Decision Log`.
4. Run the smallest relevant validation.
5. Fix breakages before expanding scope.

If blocked:
- Document the blocker in `Surprises & Discoveries`.
- Narrow scope rather than silently broadening the implementation.
- Prefer preserving current behavior over speculative refactors.

## Notes for commit / PR summary

When implementation is complete, summarize:
- what emotional phase detection rule was implemented
- which defaults were chosen and why
- where the analysis-engine change log entry was added
- whether TXT golden output changed
- whether export parity was preserved
- what validations were run

## NRC desktop implementation (2026-11-07)

Approved: external NRC ZIP/XLSX, eight categories, same chunks, exact ties as equal color bands, separate emotion PNG/TXT/CSV, marker and sentence-weighted shares. No negation/context inference. Original sentiment outputs and PY remain unchanged. Windows onefile and all affected smoke checks required.

Progress: [x] engine/import/reports; [x] desktop integration; [x] Windows build and verification.

Decisions: openpyxl desktop-only; external path in QSettings; translated duplicate lemmas union categories; longest phrase match; zero markers distinct from neutrality.

NRC validation progress: real archive imported (3837 normalized entries); artificial counts/ties/phrases and report generation passed; real literary input produced all artifacts; desktop wide/stacked and slider passed; GUI worker/tab refresh passed; web golden unchanged. Added opt-in packaged worker/UI smoke. Desktop uses Agg for safe rendering in QThread. Windows UNC build environment moved to LOCALAPPDATA after reproducing site-packages discovery failure. Final Windows build pending.

Final NRC outcome (2026-11-07): Windows onefile dist/SentiSoft.exe rebuilt after palette correction. tools/smoke_frozen.ps1 passed on the final executable with NRC and without NRC; both matched the previous tone golden after CRLF normalization. Frozen worker/UI generated required files and screenshots, inspected on Windows and offscreen. Desktop counts/import, GUI refresh, wide/stacked layout and chunk slider passed; reanalysis passed on isolated rerun; standalone export passed. Full `GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify` passed. Scientific evaluation against the colleagues' manual annotation remains a user research step; no accuracy claim is made.

## Plot usability corrections
Approved: explanatory legend for all plotted lines/backgrounds; desktop chapter-label stride (ordinal chapter order, default 1); numeric chunk input synchronized with existing file-dependent slider. Preserve scoring/TXT. Mirror plot changes and stride in exported PY. Validate controls, plots, export, golden and rebuilt Windows exe.
Progress: complete. Numeric keyboard entry and slider synchronization, file-dependent ranges, dense 240-chapter plot (stride 20 gives 12 labels), all 12 legend entries and exported CHAPTER_LABEL_STEP default passed. Layout and GUI render inspected. Standalone export smoke passed; full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed with unchanged TXT. Rebuilt dist/SentiSoft.exe passed real Windows worker/UI checks with and without external NRC.

## NRC curve-clipped fills (2026-11-07)
Approved: fill each existing emotion chunk only from y=-1 to the unchanged smoothed tone curve; use identical opaque colors in fills/legend; exact ties use a bottom-to-top gradient through all tied colors. Keep all chunk boundaries and hatch unknown chunks under the curve. Explain the black line and smoothing window on the plot. Counts, chunks, shares and TXT/CSV stay unchanged.
Progress: curve-clipped fills, matching palette and gradients implemented. Rendered pixel checks, NRC counts/import, real Chapter_7_Karenina.txt (94 sentences, chunk 3) and offscreen GUI passed; real chapter image inspected. Full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed with unchanged golden. Windows onefile rebuilt successfully; tools/smoke_frozen.ps1 -NrcPath <external ZIP> -ReportName frozen-curve-fills passed on the actual exe, including worker/UI, NRC artifacts and unchanged tone golden. Windows screenshot inspected. Synthetic TXT/CSV also match the previous renderer byte for byte. Outcome: requested rendering complete; scoring and report contracts preserved.
Decision: fixed tone axis [-1, 1], deterministic category order for vertical gradients; no interpolation between neighboring chunks or change of smoothing algorithm.
Validation: synthetic clipping/colors/ties/unknown checks, Chapter_7_Karenina.txt render, emotion GUI smoke, full verify, Windows build and frozen NRC smoke.

## Explainable emotional clusters (approved follow-up)

Purpose: preserve NRC counts and tone while making ties readable and exposing the exact marker occurrences behind sentence and chunk leaders. User chose retained gradients plus explicit pure-color designations.

Implementation source of truth:
- Extend the existing marker traversal with original sentence offsets, lemma sequences and category labels; aggregate chunks from sentence counts. Keep longest/non-overlapping matching, duplicate normalization, ties and shares unchanged.
- Add escaped offline HTML (`html` path beside png/txt/csv), sentence details appended to NRC TXT, and a QTextBrowser tab "Маркеры кластеров". Explain sentence leaders separately from chunk leaders. Clear details on new run, failure and tone-only analysis.
- Keep gradients under the curve; add an aligned eight-row leader matrix with named rows, opaque category colors and hatched unknown columns. Remove smoothing-window text from PNG but preserve the curve and its legend explanation.
- Slider initially offers 1..10, then tens and final sentence count; retain defaults and numeric arbitrary input. Rename user-facing NRC feature "Эмоциональные кластеры" while preserving technical names and keeping original tonal phases named "Эмоциональные фазы".
- Update README, NRC guide, overview, AGENTS and change logs. After implementation validation, write a standalone 4–6 paragraph popular-science article description in docs/article_emotional_clusters.md without modifying existing article material.

Progress: [x] marker/report engine; [x] GUI and plot/slider; [x] final verification and Windows exe; [x] documentation/article. Final delivery: push the verified commits to main after committing this record.
Surprises & Discoveries: the displayed minimum 3 came from manually inserting 3 in a slider originally starting at 10; numeric input already allowed 1. Existing GUI tests assume exactly three artifacts and must now expect HTML too.
Decision Log: the color designation is an eight-row matrix sharing the sentence X axis; each row has its emotion name, and every tied leader is filled. HTML is escaped and self-contained, no external resources/scripts. TXT summary and CSV columns stay stable except user-facing NRC naming; append sentence details to TXT. Internal file names/settings and original tone output remain compatible.
Validation: NRC fixtures (ties/none/phrases/repeats/multi-labels), previous-count parity, HTML escaping and offsets, GUI refresh/clear, chunk 1/2/3 and empty controls, matrix render including dense inputs, actual Chapter_7_Karenina.txt, layout; full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify; sequential export/reanalysis checks; rebuild Windows onefile and test frozen with/without NRC. Commit per logical block and push main after checks. External NRC and build artifacts stay untracked/ignored.
Outcomes & Retrospective: pending.

Marker milestone: smoke_emotions passed with real 3837-entry archive and synthetic offset, count aggregation, repetition, phrase, multi-label, tie, unknown and escaped HTML fixtures. Integration checks pending.

Build discovery: PyInstaller reached final EXE assembly but WinError 5 prevented replacing dist/SentiSoft.exe. Inspecting running processes to confirm an active executable lock; do not terminate the user application. Build to a local NTFS staging directory and deliver dist/SentiSoft-clusters.exe instead, preserving the open old file. Add optional -ExePath to the frozen helper so the delivered versioned binary is tested directly. Document the alternate file/command.

UI milestone: NRC GUI including HTML-save dispatch, tone-only and error clearing passed; slider 1/2/3, numeric sync, sentence counts 1/7/50/94/237 and layout checks passed. Main chapter/matrix and marker-tab renders inspected. Dense 240-chunk render passed. Actual prior implementation count/share parity passed at sizes 1/2/3/50. Running old exe lock confirmed by Win32_Process paths; build redirected to NTFS staging, delivery filename SentiSoft-clusters.exe.

Final validation: GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed after the last helper change, with unchanged golden. NRC trace/import/report checks, fill/matrix pixels, slider and layout, GUI HTML saving and clearing passed. Export and reanalysis smoke passed sequentially. The final Windows build staged on NTFS completed and was copied to dist/SentiSoft-clusters.exe. tools/smoke_frozen.ps1 -ExePath <that binary> passed with NRC (all four files, actual QTextBrowser contents and marker screenshot) and without NRC (original tone-only TXT matches golden after CRLF normalization). Source and Windows screenshots inspected. Documentation local links and historical anchors checked.
Outcomes & Retrospective: requested feature is implemented; counts and shares retained, markers are traceable, gradients have explicit named color rows, smoothing caption removed, minimum 1 available immediately. Documentation and six-paragraph article draft describe lexical associations and limitations without claiming measured literary accuracy. Manual assessment of the selected small passage remains with the authors. The user's open old exe remains untouched; use SentiSoft-clusters.exe for this version.

## Built-in RusEmoLex and emotion density (2026-11-08)

User-approved source of intent: replace external NRC with the original RusEmoLex.csv at commit b485f34def81f622295b28bd7e5558e3581a8a9e, automatically analyze anger/sadness/joy/fear/surprise and default desktop chunks to 1. Preserve RuSentiLex scoring, phases and standalone tone export. /PLANS.md does not exist; this repository PLANS.md is the active plan.

Progress:
- [x] Audit current engine, worker, GUI, packaging and smoke contracts.
- [x] Vendor exact CSV with SHA256 and source/publication metadata; five-category normalization, sentence/chunk counts and densities, reports and two plots.
- [x] Always-on desktop integration, density tab and saving, Russian labels, disabled pre-file controls, default 1, version/build date.
- [x] Update targeted checks; validate fixtures, chapter sizes 1/2/3, dense plots, GUI refresh/error/save, sequential tone export/reanalysis and full verify.
- [x] Update README, guide, overview, analysis/project logs, AGENTS and article text.
- [x] Build SentiSoft-RusEmoLex.exe on local NTFS; inspect bundled resources, verify real frozen offline worker/UI and tone golden; commit logical blocks and push.

Surprises & Discoveries: current build helper kills SentiSoft and removes all dist outputs, violating the approved release requirement. Replace this with isolated NTFS staging and copy only the new named executable. Current engine traces original marker offsets but retains only counts on sentences; add words and unrounded density vectors. Existing GUI tests assume optional NRC and four artifacts, so migrate their contracts.

Decision Log: preserve longest normalized non-overlapping matches, union duplicate categories, and the existing lexical tokenizer. Density = category markers / all alphabetic tokens (zero for empty); UI per 100 words, CSV decimal ratios. Choose leaders from unrounded densities; preserve exact ties. Keep existing marker/text shares separately named. Vendor provenance beside CSV and package both; no external dictionary setting is read. Density curves are stepwise over sentence coordinates. Existing emotion filename base stays; add _emotion_density.png and _emotion_sentences.csv.

Validation: all milestones below passed; final evidence follows. Milestones: engine → smoke_emotions/fills; desktop → slider/layout/GUI renders; integration → sequential export/reanalysis and make verify; release → frozen smoke and resource inspection.

Outcomes & Retrospective: all requested implementation, checks, documentation and named Windows release completed. Logical commits/push are the final delivery step.

RusEmoLex engine milestone: exact 1024-row source imported to 999 normalized keys; all five classes, real examples (including «без ума» / «испугался»), duplicate union, repeats, longest matching, ties, unequal lengths, zero denominator, six reports/plots and stair values passed smoke_emotions. Curve fill/matrix pixels passed for five colors. Fixed event-filter installation order after the new preview exposed an early Qt callback; startup/slider and layout checks now pass. Threaded GUI and release checks in progress.

RusEmoLex desktop/integration milestone: startup minimum/unknown maximum and chunk=1 defaults, keyboard synchronization, wide/stacked layout passed. Real QThread flow saved all six files byte-for-byte, refreshed them, removed partial/stale emotion files after a deliberate failure, ignored invalid saved NRC path, analyzed supplied 94-sentence chapter at sizes 1/2/3 and dense 240-sentence text. Source screenshots inspected. Test runner QTest.qWait blocked delivery of worker thread events; using a normal QEventLoop fixed the harness (no runtime workaround). Sequential standalone export and reanalysis, dense chapter/legend/export options passed. Full container HTTP make verify passed; repeated only after a final density-axis refinement to use integer sentence ticks. Windows initial frozen offline check passed with exact tone golden, archive inventory confirmed all four dictionary/provenance files plus build metadata. Final axis-refined binary is rebuilding. Added preserved-byte .gitattributes rule for the upstream CSV and updated README/guide/overview/AGENTS, both analysis change logs and the six-paragraph article text.

Final RusEmoLex validation (2026-11-08): final `.venv/bin/python desktop/smoke_emotions.py`, smoke_emotion_fills, offscreen chunk slider/layout and actual threaded GUI passed (including real six-file copies, default/reset, error deletion, isolated obsolete NRC setting, chapter sizes 1/2/3 and dense input). Export then reanalysis were run sequentially and passed; smoke_plot_options passed. After the last runtime change, `GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify` passed: precision on host/container and golden HTTP upload/download identical to baseline. Published host-port reachability is not established by container HTTP mode. Golden baseline and tone runtime/export sources remain unchanged. Final Windows onefile rebuilt from this source on local NTFS, delivered as dist/SentiSoft-RusEmoLex.exe; `tools/smoke_frozen.ps1 -ReportName rusemolex-final` passed with outgoing Python socket connections forbidden, no external dictionary, six emotion outputs, report/cluster/density screenshots and unchanged tone golden after CRLF normalization. Archive inventory confirms original CSV, SHA256, provenance/upstream README and actual build metadata. Source chapter/dense graphs and GUI, plus final Windows screenshots, visually inspected. Local documentation links and diff whitespace checked. Original user files and previous exe preserved; generated sections.txt restored before committing. Runtime is complete; no accuracy claim or external NRC data enters Git. Test settings now live in a temporary isolated directory.

Delivery record: engine/resources/checks committed as 6e8eb36; desktop/UI/packaging/checks as 0a73f5c. Final release SHA256: 6c78a41835a0330a8cbf1d427cdcc8a8bd474e39c4b0e5d94e4fc8ff432d2adb (127814984 bytes). Exact committed dictionary SHA256 verified from Git index. CSV attributes preserve CRLF and validate it as a line ending; upstream README is preserved verbatim. Final docs/screenshots and this execution record form the documentation commit; all three commits are delivered together to origin/main. No user-supplied untracked files or generated result artifacts are included.

## NRC EmoLex Russian 2022 research build (2026-11-08)

Intent: replace the desktop RusEmoLex resource with NRC EmoLex 0.92, Russian Google Translate August 2022, bundled in the user's local non-commercial research executable. User explicitly clarified no distribution. Preserve RuSentiLex/phase/standalone tone contracts, density denominator/units, traceable reports, chunk default 1, six artifacts and safe staging of Windows builds. Use all eight native NRC emotion categories; positive/negative metadata does not replace tone scoring.

Progress: [x] audit current source and official translation statement; [x] acquire/identify/hash original 2022 Russian resource; [x] loader/eight-class UI/report integration and checks; [x] docs/provenance; [x] full verify and frozen Windows release.

Surprises & Discoveries: official English base is manually crowdsourced; Russian translations are automatic Google Translate, August 2022. The old locally available suite includes 2017 XLSX; do not silently reuse it. Current renderer supports dynamic category counts, but summarize retains an eight-element allocation despite five-class mode; use len(KEYS) consistently. Current spec packages the whole data directory; narrow it to the selected NRC files so the research exe contains only its active resource. Data must be ignored by Git and Docker context; source metadata/loader/checks can be versioned without the dictionary itself.

Decision Log: keep existing five category indices/colors and append anticipation, disgust, trust, yielding anger/sadness/joy/fear/surprise/anticipation/disgust/trust. Union category labels across normalized translations and count each matched occurrence once per category. Ignore polarity labels for this feature. Load bundled UTF-8 text without openpyxl or a network dependency. Validate exact original source hash and provenance; no download at application startup. Name the new executable SentiSoft-NRC2022.exe, preserving every previous exe/process. Tests must cover real multi-label rows, normalization union, incomplete/tampered resource rejection, eight-density vectors and report dimensions. Do not calibrate counts or select words to force a sad literary outcome.

Validation: engine/fill fixtures; offscreen GUI/slider/layout, chapter 1/2/3 and dense text, full-novel count comparison; sequential export/reanalysis; GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify; Windows offline worker/UI, resources/metadata and unchanged tone golden. Outcomes & Retrospective: implementation and validation complete; local research exe retained, dictionary/README excluded from Git and web image. The resource switch is not evidence of better literary accuracy.

NRC 2022 final evidence (2026-11-08): official ZIP SHA256 4edcbd00b1d38ace19ecebca17a50ce08719785b9476cd12d31532eff0fc34e6, Russian TXT SHA256 96849cb99a395e1e749030fa6c115860e2664f7690a1beac82c47485d42ce75a; 14154 raw rows / 3707 normalized keys, eight classes. Pinned preparer validated local archive and wrote only ignored resources. smoke_emotions/fills, offscreen chunk slider/layout and real threaded GUI passed, including missing/tampered source, real multi-label records/union, eight-vector counts/densities/CSV, no matches, phrase/repeat/tie/zero/unequal lengths, escaping/PNG/stairs, six-file saving, reanalysis/error deletion, chapter sizes 1/2/3 and dense input. An initial test-only name collision between unittest.mock.patch and a stair artist variable was corrected; the suite then passed. Standalone export and reanalysis ran sequentially and passed, as did plot options. Full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed on final runtime with unchanged tone golden and host/container precision. Host published-port reachability remains unverified by container mode.

Windows build via build_windows.ps1 -OneFile completed on local NTFS as dist/SentiSoft-NRC2022.exe; smoke_frozen.ps1 -ReportName nrc2022-final passed on the real exe with outgoing Python socket connects prohibited, eight categories, six outputs and unchanged tone golden after CRLF normalization. Archive inventory contains original Russian TXT/README, pinned JSON/MD/SHA256 and actual build metadata, without RusEmoLex data. Final exe size 127961126 bytes; SHA256 9a142296ef31063ac6eca71378ac6211da92ba58f858d9936105545f7fe99fec. Source/Windows screenshots and chapter plots inspected. docker run against the verified web image confirmed scripts/data/nrc2022 does not exist; git check-ignore and git ls-files confirm raw data/README are not versioned. Current README/guide/overview/article/AGENTS and both engine logs updated, old release kept historical. Local links/JSON/whitespace checked. Source changes can be committed/pushed without the private dictionary or executable.

Full-novel exploration: on check_results/Karenina.txt (20672 sentences), NRC2022 totals are anger 5968, sadness 6938, joy 8028, fear 6565, surprise 5071, anticipation 9629, disgust 5133, trust 18332. All 20 chunks at 1050 select trust. The original row English wot→Russian что has trust=1, accounting for 6393 occurrences. This is an upstream translation artifact, not a regression of counting; no hidden stop-list/filter was added. This limitation was explained to the user during work and documented for any downstream interpretation.

Source delivery: implementation/preparation/verification changes committed as 80793db. Documentation, explicit translation-quality limitation, screenshots and release manifest form the companion documentation commit. Only source/provenance and program outputs are versioned; the original NRC TXT/README and local research exe remain ignored. Before source delivery the staged file list was checked for private resource paths and archives/executables; none were present.

## Conservative Russian NRC literary profile (2026-11-08)

User-approved intent: adapt the Russian NRC 2022 resource cautiously for general literary prose, not Anna Karenina. Correct explicit translation errors and suppress non-emotional function-word matches; preserve all other upstream emotion labels. Original NRC bytes/hash and raw mode must remain reproducible. Do not use Anna Karenina frequencies or desired plot color as a selection/training criterion.

Progress: [x] resource-wide linguistic audit with documented candidates; [x] versioned correction layer and raw/corrected loading, meaningful regression fixtures; [x] desktop/report provenance and distinct research exe; [x] general prose fixtures, GUI/export/golden/frozen verification; [x] docs and release evidence.

Decision Log: audit the entire source, using Russian morphology and English source meaning, with independent short prose/grammar fixtures. Keep multiword expressions intact when they contain prepositions/particles; do not indiscriminately strip function tokens from text or the density denominator. Do not infer negation, change scoring weights, rebalance emotion categories or reclassify valid content words. Every applied change must have a stable rule/entry identity, reason and reproducible profile digest. Original local data stay ignored by Git; only source correction rules, rationale and code are versioned. Separate current corrected exe filename from the raw NRC2022 executable, preserving open apps. Outcomes & Retrospective: conservative correction layer and original-mode comparison implemented and validated for general literary prose; no novel-based fitting, full retranslation or measured literary-accuracy improvement is claimed.

Validation: fail-first filter/override fixtures; actual raw/profile source comparison; independent literary-style and grammatical minimal examples (no fitting to a particular novel); sentence/chunk/density/report checks; existing GUI/slider/layout, sequential export/reanalysis, full verify; actual offline Windows worker/UI and metadata/profile integrity.

Literary v1 decisions: user selected cautious editing and explicitly general literary scope. Source-wide morphology audit selected 17 service/pronominal candidate rows; semantic/grammar review retained ten ambiguous content readings. Seven decisions: exclude aga/wot; replace spite→злоба and nothingness→небытие; suppress single-token translations против (opposed/versus) and исключая (excluding). The closed whole-token policy preserves expressions and all text/denominator words. Five source rows excluded, two narrowed/corrected; all other rows/tags unchanged. Raw keys 3707, literary keys 3702. Profile nrc-ru-literary-v1, canonical JSON SHA256 cbd2607a69f63549552956dd71ffb56f2ba99b76531646438499edae274a29e7. Source SHA256 unchanged at 96849cb99a395e1e749030fa6c115860e2664f7690a1beac82c47485d42ce75a. All decisions, primary dictionary references, deferred ambiguities and tradeoffs are in docs/lexicons/nrc_ru_literary_v1.md. Reproducible read-only tools/audit_nrc_profile.py takes no corpus and returns JSON: 14154 source rows, 17 candidates, seven changes, raw/profile key counts and fingerprints. User's in-progress question about full retranslation was answered: selective justified edits are useful now, a full new translation has no demonstrated accuracy advantage without independent evaluation; original agreed scope retained.

Validation: first smoke_emotion_profile failed for the missing helper, then passed all seven decisions, untouched original/other rows/tags, source-row guards, raw/default edited, independent general grammar/prose, preserved ambiguous readings/interjections, expressions/denominator and all report provenance. An initial test assumed raw negated/unnegated sentences have identical counts; inspection revealed existing не быть phrase matching, so the test now preserves raw behavior and separately tests absence of new negation inference on a simple lexicon. No runtime negation rule changed. During final review a focused GUI test demonstrated the selector could re-enable between result delivery and thread cleanup; fixed only its enable timing, reran GUI and rebuilt/reverified final runtime. Counts/fills, offscreen slider/layout, actual threaded GUI with raw↔edited switching/disabled controls/save/refresh/error, standard chapter 1/2/3 and dense render passed. Standard chapter fixtures are mechanical regressions only, not tuning/evaluation inputs. No full-novel frequencies or expected colors used. Sequential export/reanalysis and plot options passed; final GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed with unchanged tone golden and host/container precision. Published host-port reachability remains unverified by container mode. New read-only audit CLI ran and JSON parsed successfully.

Release: separate NTFS Windows build delivered dist/SentiSoft-NRC2022-Literary.exe, 127964536 bytes, SHA256 54fed1d02358483eb76a95493c31ac373b7a7247ff3e9989285538e576465e51. Both final offline frozen modes are checked using literary-final and literary-raw, expected profile fingerprints, eight categories, six files and unchanged tone golden after CRLF normalization; outbound Python socket connects forbidden. Archive contains original Russian TXT, NRC2022.LITERARY.json and build metadata. Original raw/RusEmoLex/older executables and open user apps retained. Source/Windows snapshots and reports inspected, private dictionary remains Git-ignored. README, guide, overview, AGENTS, engine logs, article, decision journal and release manifest updated; local links/JSON/whitespace checked. Generated sections.txt restored before committing. Source/documentation delivery excludes original NRC data and executables.

Final Windows visual correction: explicitly set the new profile-label foreground/background after observing a white inherited label on the light panel under the Windows dark palette. Re-ran the affected GUI and full verify and rebuilt the final exe before the final two-mode frozen checks. No dictionary or metric rule changed.

Source delivery: implementation/rules/audit and checks committed as aad41ad. Companion documentation includes the complete semantic/grammatical decision journal, deferred candidates, reproducibility instructions, screenshots and final release manifest. Both staged lists exclude the private NRC table/README and executables; those remain local. Public source delivery does not add the original licensed dataset.

## Deeper general-prose NRC profile v2 (2026-11-08)

Intent: user requested another, deeper pass for the same general literary-prose purpose. Review content-word polysemy, translation collisions and English sense evidence across the source, without novel frequencies/desired emotional plots. Preserve raw and frozen v1; create explicit v2 with documented source-row guards and independent prose fixtures. Avoid automatic relabeling, context/negation inference or broad suppression of meaningful vocabulary.

Progress: [x] source-wide collision/content audit and evidence-backed decisions; [x] v2 rules/three-profile integration and fail-first regressions; [x] audit/docs/provenance; [x] GUI/export/golden and new versioned local Windows exe in all three modes.

Decision Log: keep v1 JSON and its fingerprint immutable. Alias literary becomes v2, literary_v1 selects v1, raw stays original. Profile v2 is cumulative, preserves upstream emotion labels of retained rows, and distinguishes exclusions (unresolvable cross-language ambiguity) from narrowed/repaired translations. Every change references English/russian source and reason; no external raw NRC data or exe enters Git. New exe name must retain previous running builds.

Surprises & Discoveries: the original English sense-level table in the pinned official ZIP is available and prevents misleading replacements: blue--blueness is truly a color association; tender has offer/present/invite rather than affectionate tender; execution and treat aggregate different senses. Full-source normalized collision scan found 549 groups, 432 with differing tags (candidates, not proven errors), and four comma lists. Existing morphology conflates гора/горе: documented/deferred rather than silently changing normalization or removing grief vocabulary.

Audit outcome: 21 new decisions (17 translation replacements including suddenly alternatives, four exclusions); cumulative v1+v2 28 source rows, nine excluded/19 retranslated; keys raw3707/v1 3702/v2 3693. Retained labels untouched. English sense table SHA256 6a17bbbbbb3a84fd966506b2f5e3f693380a7e92241929ac0e26139780bffc9d is evidence only, not bundled/committed. Rules id nrc-ru-literary-v2, canonical SHA256 119bc130f86078a77053ddf1add6445cfd956e8ceac685fba0c2a9f95b311f6a. V1 remains byte-identical with its old digest. Standalone audit supports optional --archive to verify evidence and has no corpus argument. Detailed 21-decision rationale/deferred cases in docs/lexicons/nrc_ru_literary_v2.md. Source GUI visually inspected, default v2 plus v1/raw switching and guards passed; full integration and Windows evidence follows.

Outcomes & Retrospective: implementation and release validation complete. Original v1/raw retained, deeper general-prose v2 defaults in a new local exe. No claim of measured literary accuracy or exhaustive human retranslation.

V2 final evidence (2026-11-08): Validation: fail-first v2 profile check then smoke_emotion_profile_v2 and immutable-v1 smoke_emotion_profile passed: 21 added/28 cumulative decisions, unchanged original/retained tags, parent fingerprint, mismatched/missing/duplicate source guards, separate alternatives, no duplicate counts, general prose and invariant denominators. Read-only audit validated pinned ZIP/sense-member SHA256. smoke_emotions/fills, offscreen slider/layout and real threaded GUI passed with v2/v1/raw switching, six saved files, refresh/error deletion, chapter sizes 1/2/3 and dense rendering. Sequential export then reanalysis, plot options and full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed: unchanged tone golden, host/container historical rounding. Container HTTP does not verify published host-port reachability. Windows onefile built on isolated NTFS; all three offline real frozen modes (v2-final/v2-v1/v2-raw) passed, original source and profile hashes verified inside bundle, six outputs and unchanged tone golden after CRLF normalization. Source/Windows screens and plots visually inspected. Original v1 executable remains byte-identical; no user process stopped.

Release: dist/SentiSoft-NRC2022-Literary-v2.exe, 127966422 bytes, SHA256 5d89926f83e7073cf7606681bb200102cc71a2f0458d809a24a548e241582a73. Profile v2 hash 119bc130f86078a77053ddf1add6445cfd956e8ceac685fba0c2a9f95b311f6a. Archive inventory verifies original Russian resource, both JSON profiles, provenance and edition metadata; no sense-level table bundled. README, guide, overview, AGENTS, article, both engine logs, complete v2 decision journal and local release manifest updated. New local Markdown targets/JSON/whitespace checked. Original v1 rules and executable verified unchanged. Staged paths checked to exclude private raw NRC data/README, archives and executables. Implementation/rules/audit/checks committed as 8f57792; companion documentation/screenshots complete the logical source delivery. Generated sections.txt is unchanged after final checks.
