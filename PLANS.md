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

Progress: [x] marker/report engine; [ ] GUI and plot/slider; [ ] final verification and Windows exe; [ ] documentation/article and push.
Surprises & Discoveries: the displayed minimum 3 came from manually inserting 3 in a slider originally starting at 10; numeric input already allowed 1. Existing GUI tests assume exactly three artifacts and must now expect HTML too.
Decision Log: the color designation is an eight-row matrix sharing the sentence X axis; each row has its emotion name, and every tied leader is filled. HTML is escaped and self-contained, no external resources/scripts. TXT summary and CSV columns stay stable except user-facing NRC naming; append sentence details to TXT. Internal file names/settings and original tone output remain compatible.
Validation: NRC fixtures (ties/none/phrases/repeats/multi-labels), previous-count parity, HTML escaping and offsets, GUI refresh/clear, chunk 1/2/3 and empty controls, matrix render including dense inputs, actual Chapter_7_Karenina.txt, layout; full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify; sequential export/reanalysis checks; rebuild Windows onefile and test frozen with/without NRC. Commit per logical block and push main after checks. External NRC and build artifacts stay untracked/ignored.
Outcomes & Retrospective: pending.

Marker milestone: smoke_emotions passed with real 3837-entry archive and synthetic offset, count aggregation, repetition, phrase, multi-label, tie, unknown and escaped HTML fixtures. Integration checks pending.
