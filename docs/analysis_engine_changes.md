# Analysis Engine Changes

This file records intentional changes to Sentilex analysis behavior in plain language.
Update it whenever `scripts/functions.py`, `scripts/export_script.py`, or analysis orchestration
changes how input text is processed, measured, plotted, exported, or written to result files.

The goal is to make engine changes understandable without reading Python code.

## 2026-11-08 - Deeper source-sense-based Russian NRC literary profile v2

Files/functions: scripts/data/NRC2022.LITERARY.v2.json; scripts/emotions.py::read_literary_profile/profile_metadata/apply_literary_profile/load_lexicon; desktop selector/build metadata/spec, Windows helpers, audit_nrc_profile and v1/v2/GUI checks. The immutable v1 JSON and original NRC data are unchanged.

Before: default literary applied seven v1 decisions, with 3702 normalized keys; GUI offered v1/raw. Content mistranslations and comma-separated translation lists remained.

After: literary defaults to cumulative v2, literary_v1 explicitly retains v1, raw retains original3707. V2 adds 17 translation replacements (including two suddenly alternatives) and four exclusions, for 28 cumulative source-row decisions (nine exclusions/19 replacement rows), 3693 keys. Source-row guards, parent digest pin and normalized category union preserve retained upstream labels and avoid duplicate occurrence counts. GUI/report provenance identifies each mode. Canonical v2 SHA256 119bc130f86078a77053ddf1add6445cfd956e8ceac685fba0c2a9f95b311f6a. CSV shape/units/artifact bases stay unchanged; parent-v1 report reproduction uses literary_v1 explicitly. New local executable is SentiSoft-NRC2022-Literary-v2.exe; prior v1 stays available.

Why/scope: user requested a deeper general-prose pass. Formal scan found 549 normalized collision groups (432 differing-tag groups), four comma lists; targeted semantic review checked original NRC English sense-level groups. Decisions include content→довольный, court→суд, pine→чахнуть, word→обещание. Source English anomalies (e.g. blue color association), labor alternatives and гора/горе morphology collision are documented/deferred. No corpus frequencies, category balancing, novel fitting, new context/negation model or label reassignment. RuSentiLex scoring, preprocessing, phase detection/chunk arithmetic and standalone tone export are unchanged. Independent literary accuracy remains unmeasured; narrower translations may reduce recall. Full rationale: docs/lexicons/nrc_ru_literary_v2.md.

Validation: fail-first v2 profile check then smoke_emotion_profile_v2 and immutable-v1 smoke_emotion_profile passed: 21 added/28 cumulative decisions, unchanged original/retained tags, parent fingerprint, mismatched/missing/duplicate source guards, separate alternatives, no duplicate counts, general prose and invariant denominators. Read-only audit validated pinned ZIP/sense-member SHA256. smoke_emotions/fills, offscreen slider/layout and real threaded GUI passed with v2/v1/raw switching, six saved files, refresh/error deletion, chapter sizes 1/2/3 and dense rendering. Sequential export then reanalysis, plot options and full GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify passed: unchanged tone golden, host/container historical rounding. Container HTTP does not verify published host-port reachability. Windows onefile built on isolated NTFS; all three offline real frozen modes (v2-final/v2-v1/v2-raw) passed, original source and profile hashes verified inside bundle, six outputs and unchanged tone golden after CRLF normalization. Source/Windows screens and plots visually inspected. Original v1 executable remains byte-identical; no user process stopped.

## 2026-11-08 - Conservative Russian literary NRC profile v1

Files/functions: scripts/data/NRC2022.LITERARY.json; scripts/emotions.py::read_literary_profile/profile_metadata/apply_literary_profile/load_lexicon/write_results; desktop profile selector/worker, build metadata/spec/helpers; smoke_emotion_profile and affected report/GUI checks. The original local NRC source remains unchanged and private.

Before: all original Russian translations were normalized and merged, including function-word/translation artifacts; only raw NRC mode was exposed. CSV had no explicit dictionary-profile identity.

After: literary (default) applies four guarded entry decisions plus a closed single-token exclusion policy before normalization. Five source rows are excluded (aga, wot, excluding, opposed, versus), and two translations narrowed/corrected (spite→злоба, nothingness→небытие). All eight emotion tags of every retained row are unchanged; other rows are byte-field-equivalent. Source-row guards check English ID, expected Russian text and expected tags. Raw source SHA stays 96849cb99a395e1e749030fa6c115860e2664f7690a1beac82c47485d42ce75a. Raw mode reproduces 3707 keys; edited mode has 3702. Rules id nrc-ru-literary-v1, canonical JSON SHA256 cbd2607a69f63549552956dd71ffb56f2ba99b76531646438499edae274a29e7. GUI offers both modes, disables the selector during work and clears results on change. TXT/HTML and PNG captions identify the profile; both CSVs append profile ID/SHA256 columns. Filename bases remain unchanged. Local exe is SentiSoft-NRC2022-Literary.exe; previous executables retained.

Why / scope: user chose cautious fixes for general artistic prose, explicitly excluding fitting to Anna Karenina. All 14154 source rows were inspected programmatically for service/pronominal candidate readings; 17 candidates reviewed, ten ambiguous valid readings retained. Decisions use source meaning/grammar, not any novel's frequencies or desired dominant category. Audit/rationale and deferred ambiguities are in docs/lexicons/nrc_ru_literary_v1.md. This is a small editorial profile, not a full translation or measured literary-accuracy improvement.

Preserved: original source bytes, unaffected translations/tags, longest non-overlapping phrase matching, repeated markers, all words in density denominators, ties, shares, segmentation/default 1 and tone RuSentiLex/phases/standalone export. Function words inside expressions are retained; no generic negation/irony inference or emotion reweighting is added. Tradeoffs of narrow replacement and suppressed homographs are explicit.

Validation: fail-first missing-profile check then smoke_emotion_profile passed (seven decisions, untouched source/other tags, mismatched-row rejection, source/raw modes, independent prose/grammar, expressions/denominators, unchanged existing negation/phrase matching and report provenance). Counts/fills, startup/keyboard slider/layout and actual threaded GUI with raw↔edited switching, saving, rerun/error cleanup, chapter 1/2/3 and dense rendering passed. Sequential tone export/reanalysis, plot options and full container HTTP verify passed with unchanged golden. Local Windows build/frozen literary mode passed; raw-mode release evidence and artifact hash are recorded in PLANS.md. No dictionary/score edits were selected from a novel's output.

## 2026-11-08 - NRC EmoLex Russian 2022 local research build

Files/functions: scripts/emotions.py (normalize_entries/load_lexicon, source_metadata, report/plot labels and summarize), desktop main/spec/build metadata, smoke checks, tools/prepare_nrc2022.py, Windows build/frozen helpers, .gitignore/.dockerignore and NRC2022 source metadata.

Before: bundled RusEmoLex, 1024 rows / 999 normalized keys, five-category vectors, five-row matrix/curves. Source occurrence counts were not used as weights.

After: pinned original Russian-NRC-EmoLex.txt from the author's 2022 ZIP, SHA256 96849cb99a395e1e749030fa6c115860e2664f7690a1beac82c47485d42ce75a. The Russian table has 14154 rows; filtering emotional entries and uniting normalized Russian translations yields 3707 keys. Eight categories: anger/sadness/joy/fear/surprise/anticipation/disgust/trust. Existing five colors/indices retained, three added; all vectors/CSV fields/curves/matrix rows now have eight categories. Binary labels count once per occurrence per category; positive/negative metadata is not used for RuSentiLex. Reports/UI identify manual English annotation versus automatic Russian Google Translate (August 2022). Original Russian TXT/README stay Git-ignored and outside Docker, packaged only in the local research exe SentiSoft-NRC2022.exe; a pinned preparer enables local setup. Previous exe/resources retained. No startup download/openpyxl required.

Why / impact: user requested testing NRC 2022 instead of RusEmoLex for a local conference research copy, with no distribution. Category counts/report columns intentionally change; original longest-match/non-overlap/normalization union/repeats, density denominator/units, leader ties, chunk default 1 and six output filename bases stay. RuSentiLex scoring, preprocessing, tone chunk arithmetic/phases/TXT/PNG and standalone PY remain unchanged. A pre-existing fixed-eight share allocation was made category-count-dependent. No hidden lexical correction or literary calibration was introduced.

Validation: pinned 14154-row source / eight classes / 3707 keys, real multi-label examples, duplicate union, missing/tampered resource rejection, synthetic phrase/repetition/tie/zero/unequal-length densities and aggregate consistency, eight-field CSV vectors, HTML escaping and two PNG/stair geometry passed; all eight palette/matrix/fill checks passed. Offscreen startup/keyboard slider/layout and actual QThread GUI passed with six-file save, rerun/error cleanup, chapter 1/2/3 and dense input. Sequential standalone export/reanalysis, plot options and full container HTTP verify passed with unchanged tone golden. Windows build/frozen offline worker/UI passed with pinned TXT/README/source, eight categories and six outputs. Archive inspection confirmed active NRC only. Docker/Git privacy checks and final release details are recorded in PLANS.md.

Known data limitation: original wot→что is labeled trust. On local Karenina.txt (20672 sentences) it creates 6393 matches; all 20 chunks at size 1050 select trust. This version exposes the raw method, not a claim of improved literary accuracy.

## 2026-11-08 - Built-in RusEmoLex and word-normalized emotion density

Files/functions: `scripts/emotions.py` (load_lexicon, normalize_entries, count_chunks, sentence_report, write_results, draw_density, draw_leader_matrix), `scripts/data/*`; desktop worker/controller/UI, dependency manifest and packaging/smoke helpers.

Before: optional external NRC ZIP/XLSX, eight translated categories, saved dictionary path, counts and marker/text shares, four emotion artifacts. Desktop defaulted to 50 sentences when available; there was no word-normalized density plot.

After: always-on built-in original RusEmoLex.csv at commit b485f34def81f622295b28bd7e5558e3581a8a9e, checksum validated before normalization; 1024 rows become 999 normalized keys with unioned categories. Order: anger, sadness, joy, fear, surprise. Source occurrence metadata is never a weight. Longest non-overlapping phrase matching, repeated occurrences and marker offsets are retained. Each sentence and chunk records all alphabetic words, five counts and unrounded count/word densities (zero if no words). Chunk counts/words sum sentence counts/words. All nonzero maximum density classes prevail; a shared denominator preserves the ranking within a chunk while supporting cross-length comparisons.

The existing gradient plot and matrix use five categories; a separate sentence-aligned stair plot displays markers per 100 words. Chunk CSV now includes five density ratios, and a new sentence CSV includes every sentence and its marker evidence. CSV uses decimal ratios with a dot; GUI/TXT/HTML/plot units are per 100 words. Marker/text shares remain separate. TXT/escaped offline HTML contain provenance, chunk/sentence metrics and plain-language marker links. Six files: _emotions.png/.txt/.csv/.html, _emotion_density.png, _emotion_sentences.csv. Desktop defaults to chunk 1 on each selection, clears stale views/actions/files on rerun/error, ignores old NRC settings and no longer requires openpyxl. Windows release name is SentiSoft-RusEmoLex.exe; stage on NTFS without stopping previous user programs.

Why / impact: implement the requested Russian lexicon and comparable density measure with traceable evidence. Emotion category/count/report contracts intentionally change. RuSentiLex scoring, preprocessing/tokenization, tonal chunk arithmetic, emotional phases, tonal TXT/PNG and standalone PY are unchanged; the tone golden is unchanged. No measured accuracy on literary texts is claimed.

Validation: smoke_emotions (1024 rows, five real classes, phrase «без ума», inflections/repeats, normalized union, ties, zero/mixed lengths, aggregate consistency, CSV/TXT/HTML escaping and PNG/stair geometry); five-color fill/matrix pixels; startup/keyboard slider/layout; real threaded GUI with six-file saving, rerun/error clearing, chapter sizes 1/2/3 and dense text; sequential tone export/reanalysis and plot-options checks. Full container HTTP verify passed with unchanged golden; the final integer-only density X-axis refinement also passed the full verify and focused/UI checks. Windows built on NTFS; actual offline worker/UI produced all six outputs and matched tone golden, and archive inspection confirmed CSV, checksum, provenance/original README and build metadata. Final rebuilt SentiSoft-RusEmoLex.exe passed offline frozen smoke with unchanged tone golden; evidence is recorded in PLANS.md.

## 2026-11-07 - Preserve historical chunk rounding across Python versions

Files/functions:
- `scripts/functions.py::calculate_sentences_sentiments`
- Mirrored `calculate_sentences_sentiments` in `scripts/export_script.py`
- `tools/smoke_chunk_precision.py`, `tools/verify.sh`, `tools/golden_e2e.sh`

Before:
- Chunk averages used builtin `sum()`. Python 3.12 changed float accumulation,
  causing 100 sentence TXT lines in the golden scenario to round differently
  from Python 3.10, including when running the unchanged HEAD engine.

After:
- Both runtime and standalone export explicitly accumulate scores left to right,
  preserving historical Python 3.10 rounding across Python versions.
- A focused runtime/export regression exercises an average that must render
  as `+0.22`, rather than `+0.21` under compensated accumulation.

Why / impact:
- Restore reproducible historical TXT output without changing lexical scores,
  chunk boundaries, phase detection, or file structure. Golden baseline unchanged.
- Docker no longer installs unused system compiler packages whose Debian
  security downloads return 404; `.dockerignore` excludes local artifacts.

Validation:
- Focused precision regression failed before the fix on Python 3.12 and passed after.
- `GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify`: pass (2026-11-07), including runtime/export precision checks on Python 3.12 / Python 3.10 and exact golden HTTP upload/download comparison.
- `.venv/bin/python desktop/smoke_export_script.py`: pass.
- `.venv/bin/python desktop/smoke_reanalysis.py`: pass (run sequentially after export).
- Local Python 3.12 full TXT equals the golden byte for byte.
- Published-port HTTP from the host failed in this environment; explicit container HTTP mode passed. This does not establish host-port reachability.

## 2026-06-16 - More readable phase presentation on PNG plots

Files:
- `scripts/functions.py`
- `scripts/export_script.py`
- `README.md`
- `PROJECT_OVERVIEW.md`
- `PLANS.md`

Changed functions:
- `format_phase_plot_summary(...)`
- `draw_emotional_phase_backgrounds(...)`
- `plot_sentiment_curve(...)`

Before:
- Emotional phases were drawn on PNG plots as very soft background spans with low opacity.
- On dense real-text plots, the phase colors were difficult to distinguish from the gray mean corridor and section markers.
- The PNG contained only the original graph metrics at the bottom; phase details were available in TXT only.

After:
- Phase background colors are stronger, and their opacity is increased while keeping the sentiment curves readable.
- The gray mean corridor and section marker lines are visually de-emphasized so they do not wash out phase colors.
- PNG output now adds a compact phase summary below the existing graph metrics: total phase count, positive/neutral/negative shares, and average phase length.
- The same defaults and summary rendering are mirrored into exported standalone scripts.

Why:
- User review of a large desktop plot showed that the phase backgrounds were technically present but not readable enough for practical interpretation.
- A compact PNG summary makes the chart self-contained while preserving the detailed phase metrics and intervals in TXT.

Behavior impact:
- Sentence-level scoring did not change.
- Chunk aggregation did not change.
- Phase detection and phase boundaries did not change.
- TXT output structure did not change.
- PNG visual output changed intentionally.
- Export-script PNG output changed intentionally to preserve parity.

Validation:
- `python -m py_compile scripts/functions.py scripts/export_script.py`: pass.
- Direct PNG smoke check wrote `/tmp/sentilex_phase_visibility_smoke.png`: pass; visual inspection confirmed stronger phase backgrounds and the new compact phase summary.
- `.venv/bin/python desktop/smoke_export_script.py`: pass.
- `.venv/bin/python desktop/smoke_reanalysis.py`: pass.
- `make verify`: pass on 2026-06-16. Python compileall and golden e2e regression completed; downloaded web TXT matched `results/chast_7_result.txt`, so no baseline update was needed.

## 2026-06-16 - Export parity for emotional phases

Files:
- `scripts/export_script.py`
- `README.md`
- `PROJECT_OVERVIEW.md`

Changed functions:
- `format_phase_metric_lines(...)` in the export template
- `format_phase_span_lines(...)` in the export template
- `classify_phase_points(...)` in the export template
- `merge_short_phase_runs(...)` in the export template
- `build_emotional_phases(...)` in the export template
- `calculate_phase_metrics(...)` in the export template
- `draw_emotional_phase_backgrounds(...)` in the export template
- `plot_sentiment_curve(...)` in the export template
- `save_results_to_text(...)` in the export template

Before:
- The main analysis engine generated phase-aware PNG/TXT results.
- The exported standalone Python script still generated the older PNG/TXT format without emotional phase backgrounds, phase metrics, or phase intervals.

After:
- The exported standalone script contains the same adaptive phase helper logic and metric names as the main engine.
- Exported PNG output draws the same emotional phase background spans while preserving the existing editable plot customization constants.
- Exported TXT output includes the same `Эмоциональные фазы` and `Интервалы фаз` blocks as the main engine.

Why:
- The feature plan requires exported standalone analysis to reproduce the main application behavior for the same input text and chunk size.
- Desktop users rely on the exported `.py` file for local or notebook reruns, so it must not lag behind the main phase-aware output.

Behavior impact:
- Sentence-level scoring did not change.
- Chunk aggregation did not change.
- Main web/desktop TXT output did not change in this milestone.
- Export-script output changed intentionally: generated standalone scripts now produce phase-aware TXT and PNG artifacts.

Validation:
- `python -m py_compile scripts/export_script.py`: pass.
- Generated `/tmp/sentilex_export_phase_smoke.py` and compiled it with `python -m py_compile`: pass.
- Local parity check between the main engine and exported script on `texts/chast__7.txt` with `chunk_size=50`: pass, TXT outputs matched exactly in the same Python environment.
- `.venv/bin/python desktop/smoke_export_script.py`: pass.
- `make verify`: pass on 2026-06-16. Python compileall and golden e2e regression completed; downloaded web TXT matched `results/chast_7_result.txt`, so no baseline update was needed.

## 2026-06-16 - Phase summary in TXT output

Files:
- `scripts/functions.py`
- `README.md`
- `PROJECT_OVERVIEW.md`
- `results/chast_7_result.txt`

Changed functions:
- `format_phase_metric_lines(...)`
- `format_phase_span_lines(...)`
- `save_results_to_text(...)`

Before:
- The TXT result contained sentence-by-sentence sentiment values and the existing `Метрики графика` block.
- Emotional phases were calculated internally and drawn on the PNG, but the TXT did not expose phase metrics or phase intervals.

After:
- `save_results_to_text(...)` keeps the sentence lines unchanged.
- The existing `Метрики графика` block and its three metric lines stay in place.
- A new `Эмоциональные фазы` block is appended below graph metrics when phase metrics are present.
- The new block includes aggregate phase counts, average/max phase lengths, positive/negative/neutral shares, and a compact `Интервалы фаз` list in sentence coordinates.

Why:
- The feature plan requires the TXT result to report emotional phase segmentation for reproducible comparison across texts and conference reporting.
- Keeping the new block below the old metrics preserves the existing readable output while making the phase analysis explicit.

Behavior impact:
- Sentence-level scoring did not change.
- Chunk aggregation did not change.
- TXT output structure changed intentionally: a phase-summary block is appended after the graph metrics.
- PNG visual output did not change in this milestone.
- Export-script behavior did not change yet; export parity is planned for the later export milestone.

Validation:
- `python -m py_compile scripts/functions.py`: pass.
- Direct TXT smoke check with a small synthetic curve: pass, produced graph metrics plus the new phase-summary and interval blocks.
- `make verify`: pass on 2026-06-16 after updating `results/chast_7_result.txt` for the intentional appended TXT phase block.

## 2026-06-16 - Phase backgrounds on PNG plots

Files:
- `scripts/functions.py`
- `README.md`
- `PROJECT_OVERVIEW.md`

Changed functions:
- `draw_emotional_phase_backgrounds(...)`
- `plot_sentiment_curve(...)`

Before:
- The PNG plot showed the raw RuSentiLex curve, the smoothed curve, extrema lines and annotations, the mean corridor, section markers, and the lower graph-metrics block.
- Emotional phases were calculated internally after the previous change, but they were not visible on the plot.

After:
- `plot_sentiment_curve(...)` draws emotional phase intervals as subtle backgrounds behind the existing curves.
- Positive phases use a soft green background, neutral phases use a soft gray background, and negative phases use a soft red background.
- The existing raw curve, smoothed curve, extrema annotations, mean corridor, section markers, legend, and lower metrics block remain on top and readable.

Why:
- The feature plan requires emotional phases to be visible directly on the PNG, not only available as internal metrics.
- Background spans make stable emotional regions visible without replacing or hiding the original sentiment curves.

Behavior impact:
- Sentence-level scoring did not change.
- Chunk aggregation did not change.
- Output file structure did not change: TXT content is still unchanged in this milestone.
- PNG visual output changed intentionally: phase backgrounds are now shown.
- Export-script behavior did not change yet; export parity is planned for the later export milestone.

Validation:
- `python -m py_compile scripts/functions.py`: pass.
- Direct PNG smoke check wrote `/tmp/sentilex_phase_smoke.png`: pass; visual inspection confirmed phase backgrounds remain behind the existing curves and metrics block.
- `make verify`: pass on 2026-06-16. Python compileall and golden e2e regression completed; downloaded TXT matched `results/chast_7_result.txt`, so no baseline update was needed.

## 2026-06-16 - Adaptive phase helper API

Files:
- `scripts/functions.py`

Changed functions:
- `classify_phase_points(...)`
- `merge_short_phase_runs(...)`
- `build_emotional_phases(...)`
- `calculate_phase_metrics(...)`
- `plot_sentiment_curve(...)`

Before:
- The analysis engine calculated sentence/chunk sentiment values, smoothed the curve, and returned only the existing graph metrics: volatility, path length, and mean change.
- There was no internal representation of stable positive, neutral, or negative emotional phases.

After:
- The engine can classify each point of the existing smoothed curve as `positive`, `neutral`, or `negative`.
- Neutral is defined adaptively for each text as the corridor around the smoothed-curve mean: `mean +/- 0.5 * standard deviation`.
- Neighboring points with the same label are grouped into ordered phase spans.
- Very short spans are merged into neighboring stable spans. The minimum span length is tied to the current analysis granularity: `max(2, chunk_size // 2)`.
- Each phase records its start/end indexes, type, length, mean value, minimum, and maximum.
- `plot_sentiment_curve(...)` now returns these phase spans plus aggregate phase metrics in its metrics dictionary.

Why:
- This prepares Sentilex for adaptive emotional phase segmentation without changing the existing lexical scoring model.
- The adaptive corridor avoids fixed thresholds such as `0.15` and makes phase detection depend on the emotional range of the current text.
- The short-span merge rule prevents one-off spikes from becoming noisy micro-phases.

Behavior impact:
- Sentence-level scoring did not change.
- Chunk aggregation did not change.
- Output file structure did not change in this milestone: TXT still contains the same sentence lines and graph-metrics block.
- PNG visual output did not change in this milestone.
- Export-script behavior did not change yet; export parity is planned for the later export milestone.

Validation:
- `python -m py_compile scripts/functions.py`: pass.
- Direct helper smoke check with a small synthetic curve: pass, produced ordered negative/neutral/positive phases and aggregate metrics.
- `make verify`: pass on 2026-06-16. Python compileall and golden e2e regression completed; downloaded TXT matched `results/chast_7_result.txt`, so no baseline update was needed.

## Entry Template

Use this structure for each change:

```md
## YYYY-MM-DD - Short change title

Files:
- `path/to/file.py`

Changed functions:
- `function_name(...)`

Before:
- Explain the previous behavior.

After:
- Explain the new behavior.

Why:
- Explain why the change was needed.

Behavior impact:
- State whether sentence-level scoring changed.
- State whether chunk aggregation changed.
- State whether output file structure changed.
- State whether export-script behavior changed.

Validation:
- `make verify`: pass/fail or not run with reason.
- Other relevant checks: pass/fail or not run with reason.
```

## 2026-11-07 - Optional desktop NRC emotions

Added scripts/emotions.py: external ZIP/XLSX import, normalized Russian marker matching, per-chunk counts, ties, marker shares and sentence-weighted shares, separate PNG/TXT/CSV. Existing tone scoring and formats remain unchanged. app.process_text optionally returns analysis_data for desktop without recalculating sentiment. openpyxl is desktop-only. No NRC data is bundled. Validation: desktop/smoke_emotions.py covers counts, ties, phrases, missing markers, shares and real archive import; full integration follows.

NRC final validation (2026-11-07): `.venv/bin/python desktop/smoke_emotions.py`, offscreen `desktop/smoke_emotion_gui.py`, `desktop/smoke_layout.py`, `desktop/smoke_chunk_slider.py`, sequential reanalysis/export checks, and `GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify` passed. Real Windows onefile passed `tools/smoke_frozen.ps1` both with external NRC and without NRC; existing TXT matched golden after CRLF normalization. Rendering uses Agg in the desktop worker; reviewed screenshots confirmed readable new controls under the Windows dark system palette. No NRC data was included in the bundle.

## 2026-11-07 - Plot legend, chapter label stride and numeric desktop input

Before: only two curve labels were shown, every chapter label was printed, and desktop chunk size was slider-only. After: runtime/export legends explain extrema, mean, ±σ corridor and phase backgrounds outside the axes; optional chapter_label_step thins only labels; desktop exact numeric chunk input synchronizes with the file-dependent slider. app.process_text forwards stride to plotting/export; exported PY preserves CHAPTER_LABEL_STEP. Scoring, smoothing, phase detection, chunk boundaries at existing settings and TXT structure remain unchanged. Numeric input explicitly permits every integer from 1 to sentence count. Validation: control, dense-plot, export and golden checks; Windows rebuild recorded in PLANS.md.

Plot corrections validation: full verify passed with unchanged golden; smoke_plot_options.py verified all legend entries and runtime/export default stride parity, smoke_chunk_slider.py exercised numeric keyboard entry and slider synchronization, layout and GUI smoke passed and renders were inspected. Rebuilt Windows exe passed frozen checks both with NRC and without NRC.

## 2026-11-07 - NRC fills clipped to smoothed tone

`scripts/emotions.py::write_results` now uses `draw_emotion_chunks` to fill each chunk from -1 to the existing smoothed tone curve. Previously translucent emotion bands covered the entire plot and differed from the opaque legend. Fills and legend now share the same opaque palette; tied leaders use a vertical gradient, in category order bottom to top. Unknown chunks retain hatching under the curve. The black line and its unchanged `2 * chunk_size + 1` smoothing window are explained on the image. The emotion chart uses a fixed [-1, 1] tone axis.

This requested presentation change does not change counts, chunk boundaries, shares, TXT/CSV, sentiment scoring or standalone sentiment export. Validation: rendered pixel checks in `desktop/smoke_emotion_fills.py`, NRC count/import smoke, real chapter and GUI rendering, full golden and frozen exe checks (final results recorded in PLANS.md).

## Explainable NRC marker reports

`scripts/emotions.py::count_chunks` now records each matched occurrence (original offsets, spelling, normalized phrase and categories) in the existing traversal and sums sentence counts into chunks. Before, only chunk counts were retained. Longest non-overlapping matching, category union, repetition counting, maxima, shares and sentiment scoring remain unchanged. `sentence_report` renders escaped offline HTML and readable TXT from that trace; `write_results` adds an `html` artifact and appends details to NRC TXT. CSV columns and existing tone output remain unchanged. This lets users inspect evidence and distinguish sentence leaders, chunk leaders and missing markers. Validation: smoke_emotions covers traces, sums, ties, phrases, repeated/multi-label hits, escaping and outputs; final integration checks recorded in PLANS.md.

## Readable cluster gradients and desktop marker viewer

`scripts/emotions.py::write_results` retains the clipped gradient and adds `draw_leader_matrix`: eight named rows, exact palette colors, same sentence axis and chunk boundaries, hatched unknown chunks. Removed smoothing-window text from PNG; curve computation is unchanged. NRC headings now say “Эмоциональные кластеры”. `desktop/main.py` adds a marker QTextBrowser tab and HTML save action, clears previous details on rerun/error/tone-only completion, and offers 1..10 before tens in the file-derived slider while preserving defaults. Internal paths and emotion category labels stay compatible; original tone phases remain named “Эмоциональные фазы”. Detailed HTML keeps source/summary at the end so markers appear first.

Validation: NRC smoke (trace, escaping, report outputs), rendered fill/matrix pixel smoke, GUI/save/clear smoke and layout/slider checks passed. Real Chapter_7_Karenina.txt at chunk 3 rendered and inspected; counts/shares match the pre-change implementation at chunk 1/2/3/50. Dense 240-chunk rendering checked. Full golden passed; final Windows/frozen results recorded in PLANS.md. Build encountered an open old executable; use a separately named artifact and optional -ExePath frozen check without stopping the user application.
