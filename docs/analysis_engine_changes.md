# Analysis Engine Changes

This file records intentional changes to Sentilex analysis behavior in plain language.
Update it whenever `scripts/functions.py`, `scripts/export_script.py`, or analysis orchestration
changes how input text is processed, measured, plotted, exported, or written to result files.

The goal is to make engine changes understandable without reading Python code.

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
