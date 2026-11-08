# SentiSoft RusEmoLex — 2026-11-08

Локальный выпуск: `dist/SentiSoft-RusEmoLex.exe` (127814984 байта).
SHA256: `6c78a41835a0330a8cbf1d427cdcc8a8bd474e39c4b0e5d94e4fc8ff432d2adb`.
Исполняемые файлы и внешние NRC-данные не добавляются в Git.

Встроены оригинальный RusEmoLex.csv (1024 записи, пять классов), SHA256, сведения об источнике и исходный README. Автоматический анализ эмоций, размер фрагмента 1 после выбора, график плотности на 100 слов, пятистрочная матрица, объяснение найденных слов и шесть сохраняемых файлов описаны в [руководстве](../nrc_emotions.md). Тональность RuSentiLex, фазы и автономный PY сохранены.

Проверено:

- `.venv/bin/python desktop/smoke_emotions.py`: все записи/классы, реальная лексика/фразы, нормализованные совпадения, повторения, равенства, отсутствие маркеров, плотности и суммы, CSV/TXT/HTML/PNG/ступени.
- `.venv/bin/python desktop/smoke_emotion_fills.py`: обрезка по кривой, палитра, градиенты, матрица и границы.
- Offscreen `desktop/smoke_chunk_slider.py`, `desktop/smoke_layout.py`, `desktop/smoke_emotion_gui.py`: старт, значения/клавиатура, 94-предложная Chapter_7_Karenina.txt при размерах 1/2/3, смена файлов, 240-предложный плотный текст, реальное сохранение всех шести файлов, повтор и очистка при ошибке, игнорирование старой настройки NRC в отдельном тестовом каталоге.
- Последовательно `desktop/smoke_export_script.py` и `desktop/smoke_reanalysis.py`; `desktop/smoke_plot_options.py`: PASS.
- `GOLDEN_HTTP_MODE=container DOCKER_BUILDKIT=0 make verify` после последней правки runtime: PASS, прежний golden TXT совпал, округление runtime/export прошло на host и Docker. HTTP проверен внутри контейнера; доступность опубликованного порта с host не установлена.
- Windows `tools/build_windows.ps1 -OneFile`: PASS, сборка на локальном NTFS, прежние exe и работающие пользовательские процессы сохранены.
- Windows `tools/smoke_frozen.ps1 -ReportName rusemolex-final`: PASS на итоговом exe без внешнего словаря; исходящие Python socket-соединения запрещены. Реальный worker/UI создал все шесть файлов эмоций, тональный TXT совпал с golden после CRLF-нормализации. JSON и снимки кластеров, маркеров и плотностей находятся в `dist/rusemolex-final*`.
- Инвентарь PyInstaller подтверждает встроенные CSV, SHA256, источник, upstream README и `desktop/build_info.json`. Снимки GUI и графики осмотрены; ссылки документации проверены.

Эти проверки подтверждают реализацию и сохранение тонального контракта, но не измеряют точность интерпретации художественной литературы. Контекст, отрицания и ирония методом не интерпретируются.
