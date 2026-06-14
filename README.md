# Sentilex: Анализ тональности

Проект **Sentilex** предназначен для интеллектуального анализа текста и определения эмоциональной окраски (тональности) загруженных документов. В репозитории поддерживаются web-интерфейс и desktop GUI на PySide6: оба варианта позволяют загрузить `.txt`-файл, указать параметры анализа и получить результат в виде текстового файла с разметкой и графика распределения тональности по тексту. Desktop-вариант дополнительно умеет экспортировать автономный Python-скрипт для ручной доработки визуализации и повторного запуска анализа вне приложения.

Текущий формат результата:
- текстовый файл содержит построчные оценки предложений и блок агрегированных метрик графика;
- PNG-график показывает исходную кривую, одно сглаживание (`uniform_filter1d`) и встроенный нижний блок метрик;
- desktop дополнительно сохраняет `.py`-скрипт с текущей логикой анализа и блоком параметров графика для ручного редактирования;
- разделы текста отображаются на графике как визуальные маркеры и не участвуют в разрезании чанков анализа.

## Структура проекта

```bash
ProjectX/
├── Dockerfile
├── requirements.txt
├── app.py
├── desktop/
│   ├── design_variants/     # Qt Designer .ui эскизы (A/B/C)
│   ├── ui/main.ui           # Основной desktop UI
│   ├── style.qss            # QSS стили для desktop UI
│   ├── main.py              # Desktop MVP entrypoint
│   ├── requirements.txt     # Desktop-only зависимости (PySide6)
│   ├── sentilex.spec        # PyInstaller spec для Windows сборки
│   ├── render_main.py       # Рендер main.ui -> desktop_parity.png
│   └── render_variants.py   # Рендер variant_*.ui + main.ui -> PNG
├── tools/
│   └── build_windows.ps1    # Сборка desktop .exe на Windows
├── docs/
│   └── design/              # PNG-превью desktop эскизов
├── scripts/
│   ├── functions.py
│   └── RuSentilex-2017.txt
├── templates/
│   ├── upload.html
│   └── results.html
├── uploads/       # Создаётся при запуске (для загружаемых файлов)
└── results/       # Создаётся при запуске (для выходных данных)
```

## Основные файлы
* `Dockerfile`
Описывает среду для сборки Docker-образа (установка `Python`, необходимых библиотек и копирование исходников).

* `requirements.txt`
Список `Python`-зависимостей (`Flask`, `numpy`, `pandas`, `nltk` и др.).

* `app.py`
Главный файл web-приложения `Flask`. Здесь определяются маршруты:

  * `/` для загрузки файла,

  * `/results/<filename>` для отображения результатов,

  * `/download/<filename>` для скачивания файлов,

  * а также вспомогательные функции.

* `scripts/functions.py`
Логика обработки текста, вычисления тональности, построения графика и сохранения результатов.

* `desktop/main.py`, `desktop/ui/main.ui`, `desktop/style.qss`
Desktop-приложение на `PySide6`: окно выбора файла, параметры анализа, preview результатов и визуальное оформление интерфейса.

* `templates/upload.html` и `templates/results.html`
HTML-шаблоны, используемые Flask для рендеринга страниц загрузки и результатов соответственно.

* `uploads/`
Папка, куда складываются загруженные пользователем файлы.

* `results/`
Папка, где сохраняются результаты анализа (текстовые файлы и графики).

## Скриншоты
Ниже представлены скриншоты веб-интерфейса.

### Страница загрузки файла
![Страница загрузки файла](https://s.iimg.su/s/01/5stbaIWulpRlbX2lQoCJckrW4sXJMl0PQTqsjQgj.png)

### Страница результатов
![Страница результатов](https://s.iimg.su/s/01/lwF7ZexFJpUMPIRc3mZN6SJZO3cWCGWSy4tppum1.png)

## Desktop GUI дизайн-эскизы (PySide6, без изменения логики)

Добавлены три варианта главного окна нативного GUI:
- `desktop/design_variants/variant_A.ui` (minimalism modern)
- `desktop/design_variants/variant_B.ui` (pro tool)
- `desktop/design_variants/variant_C.ui` (friendly)

Сгенерированные предпросмотры:
- `docs/design/variant_A.png`
- `docs/design/variant_B.png`
- `docs/design/variant_C.png`

Рендер PNG из `.ui`:
```bash
# если есть display
.venv/bin/python desktop/render_variants.py

# для WSL/headless
QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/render_variants.py

# альтернатива через virtual X server
xvfb-run -a .venv/bin/python desktop/render_variants.py
```

Для рендера нужен PySide6 (отдельно от Flask-части):
```bash
.venv/bin/pip install -r desktop/requirements.txt
```

## Desktop GUI MVP (локальный запуск)

Desktop-часть использует те же функции анализа, что и веб, но держит Qt-зависимости отдельно.

Desktop UX (MVP):
- главное окно стартует в безопасном размере под доступную рабочую область экрана;
- на широких экранах используется широкий рабочий layout: параметры слева, результаты справа;
- на более узких окнах layout переключается в stacked-режим, а при нехватке высоты контент прокручивается без наложения виджетов;
- вводный текст desktop-окна нейтрально описывает назначение программы, без упора на layout;
- размер фрагмента в desktop задаётся в предложениях, а максимальное значение вычисляется по выбранному `.txt` файлу;
- ползунок размера фрагмента использует округлённые шаги и при необходимости добавляет точный максимум по числу предложений файла;
- после анализа текстовый результат показывается в preview-блоке;
- preview графика кликабелен и открывается в большом модальном окне;
- после анализа desktop позволяет сохранить `TXT`, `PNG` и автономный `PY`-скрипт для Google Colab или локального повторного запуска;
- повторный анализ подряд поддерживается (результаты и preview обновляются);
- PNG-график содержит нижний блок метрик и не обрезает подписи оси X.

```bash
# web/runtime зависимости
.venv/bin/pip install -r requirements.txt

# desktop-only зависимости (не для Docker/web)
.venv/bin/pip install -r desktop/requirements.txt

# запуск desktop приложения
.venv/bin/python desktop/main.py

# non-GUI smoke: двойной прогон анализа и проверка обновления TXT/PNG
.venv/bin/python desktop/smoke_reanalysis.py

# non-GUI smoke: экспортированный PY воспроизводит TXT результата
.venv/bin/python desktop/smoke_export_script.py

# headless smoke: проверка wide/stacked layout без display
QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_layout.py

# headless smoke: проверка значений ползунка размера фрагмента
QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/smoke_chunk_slider.py
```

Рендер parity-экрана:
```bash
# если есть display
.venv/bin/python desktop/render_main.py

# для WSL/headless
QT_QPA_PLATFORM=offscreen .venv/bin/python desktop/render_main.py
```

## Windows build (.exe)

Сборку `PyInstaller` нужно выполнять **на Windows** (кросс-компиляция из Linux/WSL не поддерживается).
Для desktop сборки используйте Python `3.10` или `3.11` (из-за совместимости `pymorphy2`).

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_windows.ps1
```

Опционально `onefile`:

```powershell
powershell -ExecutionPolicy Bypass -File tools/build_windows.ps1 -OneFile
```

Примечание: `tools/build_windows.ps1` намеренно пинует `setuptools<81`, потому что `pkg_resources` удалён в новых версиях, а для текущего стека сборки/рантайма это нужно.

Где искать результаты сборки:
- `dist/SentiSoft/` для `onedir` (по умолчанию)
- `dist/SentiSoft.exe` для `onefile`

Важно: desktop приложение в режиме `.exe` пишет `results/` рядом с исполняемым файлом, поэтому каталог запуска должен быть доступен на запись.

В bundle включаются обязательные desktop-ресурсы:
- `desktop/ui/main.ui`
- `desktop/style.qss`
- `scripts/RuSentilex-2017.txt`

## Установка и запуск

1. ### Локальный запуск (без `Docker`)
   * Убедитесь, что у вас установлен `Python 3.10`+ (либо другая совместимая версия).
   * Установите зависимости:
     ```bash
     pip install -r requirements.txt
     ```
   * Запустите приложение в веб-режиме:
     ```bash
     python app.py
     ``` 
   * Перейдите в браузере по адресу http://127.0.0.1:5000
   * Запустите приложение в десктоп-режиме:
     ```bash
     python desktop.py
     ```
     Откроется окно `pywebview` с интерфейсом Sentilex. При закрытии окна Flask-сервер останавливается автоматически.

2. ### Сборка и запуск в `Docker`
   * Сборка Docker-образа:
      ```bash
      docker build -t sentilex_v10 .
      ```
      Эта команда выполнит инструкции в `Dockerfile`, соберёт образ с именем `sentilex_v10` и установит все необходимые зависимости.
   * Запуск контейнера:
      ```bash
      docker run -p 5000:5000 sentilex_v10
      ```
      Здесь:
      `-p 5000:5000` перенаправляет порт 5000 контейнера на порт 5000 хоста, чтобы к приложению можно было обратиться по адресу http://localhost:5000.
   * Использование:
      * Откройте http://localhost:5000 в браузере.
      * Загрузите .txt-файл и укажите параметры анализа (например, «Размер фрагмента»).
      * После анализа вы будете перенаправлены на страницу результатов, где сможете:
        * Скачать текстовый файл с результатами и метриками графика.
        * Скачать или просмотреть график распределения тональности.


## Примечания
* Файл `RuSentilex-2017.txt` (в папке `scripts/`) содержит словарь тонально окрашенных слов и используется для вычисления эмоциональной окраски предложений.
* Папки `uploads/` и `results/` создаются автоматически при запуске приложения, если их нет.
* При желании вы можете дополнительно кастомизировать стили в файлах `templates/upload.html` и `templates/results.html`, чтобы добиться нужного внешнего вида.

## Лицензия
Проект распространяется под лицензией MIT. Подробности в файле [LICENSE](LICENSE).
