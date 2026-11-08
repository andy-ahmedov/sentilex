#!/usr/bin/env python3
"""SentiSoft desktop MVP (PySide6) with web-parity upload UI."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QFile, QObject, QEvent, QTimer, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QPixmap, QTextCursor
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QBoxLayout,
    QFileDialog,
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTextBrowser,
    QProgressBar,
    QTabWidget,
    QWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QVBoxLayout,
)
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IS_FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", str(PROJECT_ROOT)))
RUNTIME_ROOT = Path(sys.executable).resolve().parent if IS_FROZEN else PROJECT_ROOT


def resource_path(*parts: str) -> Path:
    return RESOURCE_ROOT.joinpath(*parts)


def runtime_path(*parts: str) -> Path:
    return RUNTIME_ROOT.joinpath(*parts)


def debug_log(message: str) -> None:
    if os.environ.get("SENTILEX_DEBUG") == "1":
        print(f"[sentisoft] {message}")


def configure_frozen_dict_path() -> None:
    if not IS_FROZEN:
        return

    bundled_dict_path = resource_path("pymorphy2_dicts_ru", "data")
    os.environ.setdefault("PYMORPHY2_DICT_PATH", str(bundled_dict_path))
    configured_path = Path(os.environ["PYMORPHY2_DICT_PATH"])
    debug_log(
        "PYMORPHY2_DICT_PATH="
        f"{configured_path} (exists={configured_path.exists()})"
    )


def prepare_runtime_environment() -> None:
    try:
        runtime_path("results").mkdir(parents=True, exist_ok=True)
        runtime_path("uploads").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Cannot create runtime folders in {RUNTIME_ROOT}") from exc

    if IS_FROZEN:
        source_lexicon = resource_path("scripts", "RuSentilex-2017.txt")
        target_lexicon = runtime_path("scripts", "RuSentilex-2017.txt")
        try:
            target_lexicon.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Cannot create runtime scripts folder in {RUNTIME_ROOT}") from exc
        if not source_lexicon.exists():
            raise FileNotFoundError(f"Missing bundled lexicon: {source_lexicon}")
        try:
            shutil.copyfile(source_lexicon, target_lexicon)
        except OSError as exc:
            raise RuntimeError(f"Cannot prepare runtime lexicon in {target_lexicon.parent}") from exc


configure_frozen_dict_path()
prepare_runtime_environment()
os.chdir(RUNTIME_ROOT)
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The packaged smoke forbids outbound socket connections before importing analysis.
if os.environ.get("SENTILEX_SMOKE_OFFLINE") == "1":
    def _offline_audit(event, args):
        if event == "socket.connect":
            raise RuntimeError("Offline smoke: network connection attempted")
    sys.addaudithook(_offline_audit)

import matplotlib  # noqa: E402

# Workers render files; a GUI Matplotlib backend cannot run in QThread.
matplotlib.use("Agg")
import app as web_app  # noqa: E402


class AnalysisWorker(QObject):
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, file_path: Path, chunk_size: int, chapter_label_step=1, emotion_profile='literary') -> None:
        super().__init__()
        self.chapter_label_step = chapter_label_step
        self.emotion_profile = emotion_profile
        self.file_path = file_path
        self.chunk_size = chunk_size

    @Slot()
    def run(self) -> None:
        from scripts import emotions
        filename = secure_filename(self.file_path.name)
        emotion_paths = emotions.result_paths(runtime_path('results', filename))
        try:
            # Never expose artifacts from a previous or incomplete analysis.
            for path in emotion_paths.values():
                Path(path).unlink(missing_ok=True)
            lexicon, digest = emotions.load_lexicon(web_app.functions.morph_vocab, profile=self.emotion_profile)
            artifacts = web_app.process_text(
                str(self.file_path),
                filename,
                self.chunk_size,
                export_script_file=True,
                chapter_label_step=self.chapter_label_step,
                include_analysis_data=True,
            )

            result_txt = Path(artifacts["result_txt"])
            result_png = Path(artifacts["result_png"])
            result_py = Path(artifacts["result_py"])

            if not result_txt.exists() or not result_png.exists() or not result_py.exists():
                raise FileNotFoundError("Файлы результатов не были созданы")

            data = artifacts.pop("analysis_data")
            artifacts["emotions"] = emotions.write_results(
                data["sentences"], data["sentiments"], self.chunk_size, lexicon,
                digest, web_app.functions.morph_vocab, result_txt.parent / filename,
                profile=self.emotion_profile,
            )
            artifacts['emotion_profile'] = self.emotion_profile
            self.finished.emit(artifacts)
        except Exception as exc:  # noqa: BLE001
            for path in emotion_paths.values():
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
            self.failed.emit(str(exc))


class MainWindowController(QObject):
    RESPONSIVE_STACK_WIDTH = 1180
    WIDE_CONTROL_PANEL_WIDTH = 360

    def __init__(self, window: QMainWindow) -> None:
        super().__init__()
        self.window = window

        self.content_scroll_area = self._req(QScrollArea, "contentScrollArea")
        self.control_panel = self._req(QFrame, "controlPanel")
        self.results_panel = self._req(QFrame, "resultsPanel")
        self.upload_box = self._req(QFrame, "uploadBox")
        self.upload_label = self._req(QLabel, "uploadLabel")
        self.file_path_edit = self._req(QLineEdit, "filePathEdit")
        self.browse_button = self._req(QPushButton, "browseButton")
        self.chunk_slider = self._req(QSlider, "chunkSizeSlider")
        self.chunk_min_label = self._req(QLabel, "chunkMinLabel")
        self.chunk_max_label = self._req(QLabel, "chunkMaxLabel")
        self.chunk_current_label = self._req(QLabel, "chunkCurrentLabel")
        self.chunk_hint_label = self._req(QLabel, "chunkHintLabel")
        self.sentence_count_label = self._req(QLabel, "sentenceCountLabel")
        self.submit_button = self._req(QPushButton, "submitButton")
        self.status_label = self._req(QLabel, "statusLabel")
        self.progress_bar = self._req(QProgressBar, "progressBar")
        self.result_text_preview = self._req(QPlainTextEdit, "resultTextPreview")
        self.plot_preview = self._req(QLabel, "plotPreview")
        self.results_splitter = self._req(QSplitter, "resultsSplitter")
        self.download_txt_button = self._req(QPushButton, "downloadTxtButton")
        self.download_png_button = self._req(QPushButton, "downloadPngButton")
        self.download_py_button = self._req(QPushButton, "downloadPyButton")
        self.open_results_folder_button = self._req(QPushButton, "openResultsFolderButton")
        self.workspace_layout = self._req_layout("workspaceLayout")

        self.selected_file: Path | None = None
        self.current_result_txt: Path | None = None
        self.current_result_png: Path | None = None
        self.current_result_py: Path | None = None
        self.thread: QThread | None = None
        self.worker: AnalysisWorker | None = None
        self.analysis_in_progress = False
        self.current_plot_pixmap: QPixmap | None = None
        self.current_layout_mode: str | None = None
        self.allowed_chunk_sizes: list[int] = []
        self.selected_sentence_count = 0

        self.chunk_number = QSpinBox()
        self.chunk_number.setObjectName("chunkSizeInput")
        self.chunk_number.setKeyboardTracking(False)
        self.chunk_number.setPrefix("Размер фрагмента: ")
        self.chunk_number.setRange(1, 1)
        self.chunk_number.setEnabled(False)
        self.chapter_step = QSpinBox()
        self.chapter_step.setObjectName("chapterLabelStep")
        self.chapter_step.setRange(1, 100000)
        self.chapter_step.setPrefix("Подписывать каждую ")
        self.chapter_step.setSuffix("-ю главу")
        self.chapter_step.setToolTip("1 — все главы, 5 — главы 1, 6, 11… Только подписи, не размер анализа.")
        for layout in self.control_panel.findChildren(QBoxLayout):
            index = layout.indexOf(self.chunk_current_label)
            if index >= 0:
                layout.insertWidget(index + 1, self.chunk_number)
                layout.insertWidget(index + 2, self.chapter_step)
                break
        self.chunk_number.valueChanged.connect(self._on_chunk_number_changed)
        from scripts import emotions
        self.emotion_profile = QComboBox()
        self.emotion_profile.setObjectName("emotionDictionaryProfile")
        for key in ('literary', 'literary_v1', 'raw'):
            self.emotion_profile.addItem(emotions.PROFILE_LABELS[key], key)
        self.emotion_profile.setToolTip(
            "Правки для прозы v2: уточнённые переводы по значениям NRC. "
            "Версия v1 и исходный перевод оставлены для сравнения. После смены повторите анализ.")
        profile_label = QLabel("Профиль эмоционального словаря")
        profile_label.setStyleSheet("color:#483d65; background:transparent;")
        self.control_panel.layout().addWidget(profile_label)
        self.control_panel.layout().addWidget(self.emotion_profile)
        self.emotion_results = {}
        self.emotion_save_buttons = {}
        import json
        info = json.loads(resource_path('desktop', 'build_info.json').read_text(encoding='utf-8'))
        self.version_label = QLabel(f'NRC 2022 · правки для прозы v2 · сборка {info["build_date"]}')
        self.version_label.setWordWrap(True)
        self.version_label.setStyleSheet("color:#483d65; background:transparent;")
        self.version_label.setToolTip(
            "NRC EmoLex: восемь эмоций; доступны правки v2, v1 и исходный профиль. Saif M. Mohammad, Peter D. Turney; NRC Canada. "
            "Русский перевод: Google Translate, август 2022. Контекст, отрицания и ирония не учитываются. "
            "https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm")
        self.control_panel.layout().addWidget(self.version_label)
        self.window.setWindowTitle(f'SentiSoft · NRC2022-Literary-v2 · {info["build_date"]}')
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setStyleSheet("QTabWidget::pane {background:#faf9ff; border:1px solid #d8ccff;} QTabBar::tab {background:#eee8ff; color:#5632c3; padding:7px 14px;} QTabBar::tab:selected {background:#faf9ff; font-weight:bold;} QWidget {background:#faf9ff; color:#483d65;} QPushButton {background:#eee8ff; color:#5632c3; border:1px solid #d8ccff; border-radius:8px; padding:6px;}")
        index = self.results_splitter.indexOf(self.plot_preview)
        self.plot_preview.setParent(None)
        self.plot_tabs.addTab(self.plot_preview, "Тональность")
        emotion_page = QWidget()
        emotion_layout = QVBoxLayout(emotion_page)
        self.emotion_preview = QLabel("Выберите текст и выполните анализ")
        self.emotion_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.emotion_preview.setMinimumSize(100, 100)
        emotion_layout.addWidget(self.emotion_preview, 1)
        for kind, title in (("png", "График кластеров PNG"), ("txt", "Отчёт TXT"),
                            ("csv", "Фрагменты CSV"), ("sentences_csv", "Предложения CSV")):
            button = QPushButton("Сохранить: " + title)
            button.clicked.connect(lambda checked=False, k=kind: self.save_emotion_result(k))
            button.setEnabled(False)
            self.emotion_save_buttons[kind] = button
            emotion_layout.addWidget(button)
        self.plot_tabs.addTab(emotion_page, "Эмоциональные кластеры")
        density_page = QWidget()
        density_layout = QVBoxLayout(density_page)
        self.density_preview = QLabel("Выберите текст и выполните анализ")
        self.density_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.density_preview.setMinimumSize(100, 100)
        density_layout.addWidget(self.density_preview, 1)
        density_hint = QLabel("Число маркеров каждой эмоции на 100 слов фрагмента. Нажмите на график для увеличения.")
        density_hint.setWordWrap(True)
        density_layout.addWidget(density_hint)
        density_button = QPushButton("Сохранить плотность эмоций PNG")
        density_button.clicked.connect(lambda: self.save_emotion_result("density_png"))
        density_button.setEnabled(False)
        self.emotion_save_buttons['density_png'] = density_button
        density_layout.addWidget(density_button)
        self.plot_tabs.addTab(density_page, "Плотность эмоций")
        marker_page = QWidget()
        marker_layout = QVBoxLayout(marker_page)
        self.emotion_report = QTextBrowser()
        self.emotion_report.setObjectName("emotionMarkerReport")
        self.emotion_report.setOpenLinks(False)
        self.emotion_report.setOpenExternalLinks(False)
        self.emotion_report.setPlainText("После анализа здесь появятся найденные слова и связанные с ними эмоции.")
        marker_layout.addWidget(self.emotion_report, 1)
        self.save_cluster_html_button = QPushButton("Сохранить отчёт кластеров HTML")
        self.save_cluster_html_button.setEnabled(False)
        self.emotion_save_buttons["html"] = self.save_cluster_html_button
        self.save_cluster_html_button.clicked.connect(lambda: self.save_emotion_result("html"))
        marker_layout.addWidget(self.save_cluster_html_button)
        self.plot_tabs.addTab(marker_page, "Маркеры кластеров")
        self.results_splitter.insertWidget(index, self.plot_tabs)
        self.emotion_preview.installEventFilter(self)
        self.density_preview.installEventFilter(self)
        self.window.installEventFilter(self)
        self.upload_box.setCursor(self.browse_button.cursor())
        self.upload_box.installEventFilter(self)
        self.plot_preview.installEventFilter(self)

        self.chunk_slider.valueChanged.connect(self._on_chunk_changed)
        self.browse_button.clicked.connect(self.choose_file)
        self.emotion_profile.currentIndexChanged.connect(self._on_emotion_profile_changed)
        self.submit_button.clicked.connect(self.start_analysis)
        self.download_txt_button.clicked.connect(self.save_txt_result)
        self.download_png_button.clicked.connect(self.save_png_result)
        self.download_py_button.clicked.connect(self.save_py_result)
        self.open_results_folder_button.clicked.connect(self.open_results_folder)

        self.results_splitter.setStretchFactor(0, 3)
        self.results_splitter.setStretchFactor(1, 2)
        self._configure_chunk_slider(0)
        self._set_idle_state()
        self._on_chunk_changed(self.chunk_slider.value())
        QTimer.singleShot(0, self._post_show_setup)

    def _req(self, cls, name: str):
        widget = self.window.findChild(cls, name)
        if widget is None:
            raise RuntimeError(f"Missing widget '{name}' in UI")
        return widget

    def _req_layout(self, name: str) -> QBoxLayout:
        layout = self.window.findChild(QBoxLayout, name)
        if layout is None:
            raise RuntimeError(f"Missing layout '{name}' in UI")
        return layout

    @Slot()
    def _post_show_setup(self) -> None:
        self._apply_responsive_layout()
        self._set_initial_splitter_sizes()

    def _set_initial_splitter_sizes(self) -> None:
        total_height = max(self.results_splitter.size().height(), 420)
        plot_height = int(total_height * 0.56)
        text_height = max(total_height - plot_height, 180)
        self.results_splitter.setSizes([plot_height, text_height])

    def _apply_responsive_layout(self) -> None:
        viewport_width = self.content_scroll_area.viewport().width()
        if viewport_width <= 0:
            viewport_width = self.window.width()

        next_mode = "stacked" if viewport_width < self.RESPONSIVE_STACK_WIDTH else "wide"
        if next_mode == self.current_layout_mode:
            return

        if next_mode == "stacked":
            self.workspace_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            self.workspace_layout.setSpacing(16)
            self.control_panel.setMinimumWidth(0)
            self.control_panel.setMaximumWidth(16777215)
            control_policy = self.control_panel.sizePolicy()
            control_policy.setHorizontalPolicy(QSizePolicy.Policy.Expanding)
            self.control_panel.setSizePolicy(control_policy)
        else:
            self.workspace_layout.setDirection(QBoxLayout.Direction.LeftToRight)
            self.workspace_layout.setSpacing(20)
            self.control_panel.setMinimumWidth(self.WIDE_CONTROL_PANEL_WIDTH)
            self.control_panel.setMaximumWidth(380)
            control_policy = self.control_panel.sizePolicy()
            control_policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
            self.control_panel.setSizePolicy(control_policy)

        self.workspace_layout.setStretch(0, 0)
        self.workspace_layout.setStretch(1, 1)
        self.current_layout_mode = next_mode

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        from shiboken6 import isValid
        if not isValid(self.window):
            return False
        if watched is self.window and event.type() == QEvent.Resize:
            self._apply_responsive_layout()
        if watched is self.upload_box and event.type() == QEvent.MouseButtonRelease:
            if self.analysis_in_progress:
                return True
            self.choose_file()
            return True
        if watched in (self.emotion_preview, self.density_preview):
            if event.type() == QEvent.Resize:
                self.refresh_emotion_preview()
            if event.type() == QEvent.MouseButtonRelease and self.emotion_results:
                self.open_plot_preview(Path(self.emotion_results["density_png" if watched is self.density_preview else "png"]))
                return True
        if watched is self.plot_preview:
            if event.type() == QEvent.MouseButtonRelease:
                self.open_plot_preview()
                return True
            if event.type() == QEvent.Resize:
                self._refresh_plot_preview()
        return super().eventFilter(watched, event)

    def _set_idle_state(self) -> None:
        self.status_label.setText("Состояние: ожидание файла")
        self.progress_bar.setVisible(False)
        self.submit_button.setEnabled(False)
        self.browse_button.setEnabled(True)
        self.chunk_slider.setEnabled(False)
        self.upload_box.setEnabled(True)
        self.download_txt_button.setEnabled(False)
        self.download_png_button.setEnabled(False)
        self.download_py_button.setEnabled(False)
        self.open_results_folder_button.setEnabled(False)
        self._set_plot_click_enabled(False)

    @Slot(int)
    def _on_chunk_number_changed(self, value):
        if not self.allowed_chunk_sizes or self.analysis_in_progress:
            return
        if value not in self.allowed_chunk_sizes:
            self.allowed_chunk_sizes = sorted(self.allowed_chunk_sizes + [value])
        self.chunk_slider.blockSignals(True)
        self.chunk_slider.setMaximum(len(self.allowed_chunk_sizes) - 1)
        self.chunk_slider.setValue(self.allowed_chunk_sizes.index(value))
        self.chunk_slider.blockSignals(False)
        self.chunk_min_label.setText(str(self.allowed_chunk_sizes[0]))
        self.chunk_current_label.setText(f"Текущее значение: {value}")

    @Slot(int)
    def _on_chunk_changed(self, index: int) -> None:
        if not self.allowed_chunk_sizes:
            self.chunk_current_label.setText("Текущее значение: недоступно")
            return
        clamped_index = max(0, min(index, len(self.allowed_chunk_sizes) - 1))
        self.chunk_number.blockSignals(True)
        self.chunk_number.setValue(self.allowed_chunk_sizes[clamped_index])
        self.chunk_number.blockSignals(False)
        self.chunk_current_label.setText(
            f"Текущее значение: {self.allowed_chunk_sizes[clamped_index]}"
        )

    def _build_allowed_chunk_sizes(self, sentence_count: int) -> list[int]:
        if sentence_count <= 0:
            return []
        if sentence_count <= 10:
            return list(range(1, sentence_count + 1))

        allowed_sizes = list(range(1, 10)) + list(range(10, sentence_count + 1, 10))
        if not allowed_sizes or allowed_sizes[-1] != sentence_count:
            allowed_sizes.append(sentence_count)
        return allowed_sizes

    def _pick_default_chunk_size(self, allowed_sizes: list[int]) -> int:
        if not allowed_sizes:
            return 0
        return 1

    def _configure_chunk_slider(self, sentence_count: int) -> None:
        self.selected_sentence_count = sentence_count
        self.chunk_number.blockSignals(True)
        self.chunk_number.setRange(1, max(1, sentence_count))
        self.chunk_number.setEnabled(sentence_count > 0)
        self.chunk_number.blockSignals(False)
        self.allowed_chunk_sizes = self._build_allowed_chunk_sizes(sentence_count)

        if not self.allowed_chunk_sizes:
            self.chunk_slider.setEnabled(False)
            self.chunk_min_label.setText("1")
            self.chunk_max_label.setText("—")
            self.chunk_current_label.setText("Текущее значение: недоступно")
            self.chunk_number.setValue(1)
            self.sentence_count_label.setText("Выберите файл, чтобы определить число предложений.")
            return

        default_chunk_size = self._pick_default_chunk_size(self.allowed_chunk_sizes)
        default_index = self.allowed_chunk_sizes.index(default_chunk_size)

        self.chunk_slider.blockSignals(True)
        self.chunk_slider.setMinimum(0)
        self.chunk_slider.setMaximum(len(self.allowed_chunk_sizes) - 1)
        self.chunk_slider.setSingleStep(1)
        self.chunk_slider.setPageStep(1)
        self.chunk_slider.setValue(default_index)
        self.chunk_slider.blockSignals(False)

        self.chunk_number.blockSignals(True)
        self.chunk_number.setValue(default_chunk_size)
        self.chunk_number.blockSignals(False)
        self.chunk_slider.setEnabled(True)
        self.chunk_min_label.setText(str(self.allowed_chunk_sizes[0]))
        self.chunk_max_label.setText(str(self.allowed_chunk_sizes[-1]))
        self.chunk_current_label.setText(f"Текущее значение: {default_chunk_size}")
        self.sentence_count_label.setText(
            f"Количество предложений в выбранном файле: {sentence_count}"
        )

    @Slot()
    def choose_file(self) -> None:
        if self.analysis_in_progress:
            return
        file_name, _ = QFileDialog.getOpenFileName(
            self.window,
            "Выберите .txt файл",
            str(runtime_path()),
            "Текстовые файлы (*.txt)",
        )
        if not file_name:
            return

        file_path = Path(file_name)
        if not web_app.allowed_file(file_path.name):
            QMessageBox.warning(self.window, "Неверный формат", "Выберите файл с расширением .txt")
            return

        try:
            text = file_path.read_text(encoding="utf-8")
            processed_text = web_app.functions.preprocess_text(text)
            sentences, _ = web_app.functions.extract_sentences_and_sections(processed_text)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self.window, "Ошибка чтения файла", str(exc))
            return

        if not sentences:
            QMessageBox.warning(
                self.window,
                "Пустой результат",
                "В выбранном файле не удалось выделить предложения для анализа.",
            )
            return

        self.selected_file = file_path
        self.file_path_edit.setText(str(file_path))
        self.upload_label.setText(file_path.name)
        self._configure_chunk_slider(len(sentences))
        self.clear_results("Параметры нового файла готовы. Нажмите «Анализировать».")
        self.status_label.setText("Состояние: готово к анализу")
        self.submit_button.setEnabled(True)

    @Slot(int)
    def _on_emotion_profile_changed(self, index):
        if self.analysis_in_progress:
            return
        self.clear_results("Профиль словаря изменён. Выполните анализ заново.")
        self.status_label.setText("Состояние: готово к анализу" if self.selected_file else "Состояние: ожидание файла")

    def _set_analyzing_state(self) -> None:
        self.emotion_profile.setEnabled(False)
        self.analysis_in_progress = True
        self.chunk_number.setEnabled(False)
        self.chapter_step.setEnabled(False)
        self.status_label.setText("Состояние: идёт анализ")
        self.progress_bar.setVisible(True)
        self.submit_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.chunk_slider.setEnabled(False)
        self.upload_box.setEnabled(False)
        self.download_txt_button.setEnabled(False)
        self.download_png_button.setEnabled(False)
        self.download_py_button.setEnabled(False)
        self.open_results_folder_button.setEnabled(False)
        self._set_plot_click_enabled(False)

    def _set_done_state(self) -> None:
        self.emotion_profile.setEnabled(not self.analysis_in_progress)
        self.chunk_number.setEnabled(bool(self.allowed_chunk_sizes))
        self.chapter_step.setEnabled(True)
        self.status_label.setText("Состояние: анализ завершён")
        self.progress_bar.setVisible(False)
        self.browse_button.setEnabled(True)
        self.chunk_slider.setEnabled(bool(self.allowed_chunk_sizes))
        self.upload_box.setEnabled(True)
        self.submit_button.setEnabled(True)
        self.download_txt_button.setEnabled(True)
        self.download_png_button.setEnabled(True)
        self.download_py_button.setEnabled(True)
        self.open_results_folder_button.setEnabled(True)
        self._set_plot_click_enabled(self.current_plot_pixmap is not None)

    def _set_error_state(self, message: str) -> None:
        self.emotion_profile.setEnabled(not self.analysis_in_progress)
        self.chunk_number.setEnabled(bool(self.allowed_chunk_sizes))
        self.chapter_step.setEnabled(True)
        self.status_label.setText("Состояние: ошибка анализа")
        self.progress_bar.setVisible(False)
        self.browse_button.setEnabled(True)
        self.chunk_slider.setEnabled(bool(self.allowed_chunk_sizes))
        self.upload_box.setEnabled(True)
        self.submit_button.setEnabled(True)
        self.download_txt_button.setEnabled(False)
        self.download_png_button.setEnabled(False)
        self.download_py_button.setEnabled(False)
        self.open_results_folder_button.setEnabled(False)
        self._set_plot_click_enabled(False)
        QMessageBox.critical(self.window, "Ошибка анализа", message)

    @Slot()
    def start_analysis(self) -> None:
        if self.analysis_in_progress:
            return
        if self.selected_file is None:
            QMessageBox.warning(self.window, "Нет файла", "Сначала выберите .txt файл")
            return

        self.clear_results("Идёт анализ…")

        self._set_analyzing_state()

        if not self.allowed_chunk_sizes:
            QMessageBox.warning(self.window, "Нет параметров", "Сначала выберите корректный .txt файл")
            self._set_error_state("Недоступен размер фрагмента")
            return

        chunk_size = self.allowed_chunk_sizes[self.chunk_slider.value()]
        self.thread = QThread(self.window)
        self.worker = AnalysisWorker(self.selected_file, chunk_size, self.chapter_step.value(), self.emotion_profile.currentData())
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.failed.connect(self.on_analysis_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_worker)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    @Slot()
    def _cleanup_worker(self) -> None:
        self.analysis_in_progress = False
        self.emotion_profile.setEnabled(True)
        self.thread = None
        self.worker = None

    def _load_txt_preview(self, txt_path: Path) -> str:
        return txt_path.read_text(encoding="utf-8")

    def _load_png_pixmap(self, png_path: Path) -> QPixmap:
        png_data = png_path.read_bytes()
        pixmap = QPixmap()
        if not pixmap.loadFromData(png_data):
            raise RuntimeError("Не удалось загрузить PNG")
        return pixmap

    def _set_plot_click_enabled(self, enabled: bool) -> None:
        if enabled:
            self.plot_preview.setCursor(Qt.CursorShape.PointingHandCursor)
            self.plot_preview.setToolTip("Нажмите, чтобы открыть график в большом размере")
            return
        self.plot_preview.setCursor(Qt.CursorShape.ArrowCursor)
        self.plot_preview.setToolTip("")

    def _refresh_plot_preview(self) -> None:
        if self.current_plot_pixmap is None or self.current_plot_pixmap.isNull():
            return
        preview_size = self.plot_preview.size()
        if preview_size.width() <= 0 or preview_size.height() <= 0:
            return
        scaled = self.current_plot_pixmap.scaled(
            preview_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.plot_preview.setPixmap(scaled)
        self.plot_preview.setText("")

    @Slot(dict)
    def on_analysis_finished(self, artifacts) -> None:
        txt_path, png_path, py_path = (artifacts[k] for k in ("result_txt", "result_png", "result_py"))
        self.emotion_results = artifacts.get("emotions", {})
        try:
            self.refresh_emotion_preview()
            if self.emotion_results:
                self.emotion_report.setHtml(Path(self.emotion_results["html"]).read_text(encoding="utf-8"))
                self.emotion_report.moveCursor(QTextCursor.MoveOperation.Start)
            else:
                self.emotion_report.setPlainText("Результаты эмоций недоступны. Повторите анализ.")
            for button in self.emotion_save_buttons.values():
                button.setEnabled(bool(self.emotion_results))
            self.current_result_txt = Path(txt_path)
            self.current_result_png = Path(png_path)
            self.current_result_py = Path(py_path)

            txt_data = self._load_txt_preview(self.current_result_txt)
            self.result_text_preview.setPlainText(txt_data)
            self.result_text_preview.moveCursor(QTextCursor.MoveOperation.Start)

            self.current_plot_pixmap = self._load_png_pixmap(self.current_result_png)
            self._refresh_plot_preview()
            self._set_done_state()
        except Exception as exc:  # noqa: BLE001
            self.on_analysis_failed(str(exc))

    @Slot(str)
    def on_analysis_failed(self, message: str) -> None:
        self.clear_results("Не удалось выполнить анализ. Повторите попытку.")
        self._set_error_state(message)

    @Slot()
    def save_txt_result(self) -> None:
        if self.current_result_txt is None:
            return
        self._save_result_copy(self.current_result_txt, "Текстовые файлы (*.txt)")

    @Slot()
    def save_png_result(self) -> None:
        if self.current_result_png is None:
            return
        self._save_result_copy(self.current_result_png, "Изображения PNG (*.png)")

    @Slot()
    def save_py_result(self) -> None:
        if self.current_result_py is None:
            return
        self._save_result_copy(self.current_result_py, "Скрипт Python (*.py)")

    def _save_result_copy(self, source_path: Path, file_filter: str) -> None:
        target_path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Сохранить файл",
            str(source_path.name),
            file_filter,
        )
        if not target_path:
            return
        try:
            shutil.copyfile(source_path, target_path)
        except OSError as exc:
            QMessageBox.critical(self.window, "Ошибка сохранения", str(exc))

    @Slot()
    def open_results_folder(self) -> None:
        results_dir = runtime_path("results")
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(results_dir)))

    @Slot()
    def open_plot_preview(self, image_path=None) -> None:
        image_path = image_path or self.current_result_png
        if image_path is None or not image_path.exists():
            return

        try:
            pixmap = self._load_png_pixmap(image_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self.window, "График недоступен", str(exc))
            return

        dialog = QDialog(self.window)
        dialog.setWindowTitle(f"График: {image_path.name}")
        screen = self.window.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            dialog.resize(min(1200, available.width() - 80), min(820, available.height() - 80))
        else:
            dialog.resize(960, 680)

        layout = QVBoxLayout(dialog)
        scroll_area = QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)
        image_label = QLabel(scroll_area)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setPixmap(pixmap)
        scroll_area.setWidget(image_label)
        layout.addWidget(scroll_area)

        dialog.exec()

    def clear_results(self, message):
        self.emotion_results = {}
        self.emotion_report.setPlainText(message)
        for button in self.emotion_save_buttons.values():
            button.setEnabled(False)
        for preview in (self.emotion_preview, self.density_preview, self.plot_preview):
            preview.clear()
            preview.setText(message)
        self.current_result_txt = None
        self.current_result_png = None
        self.current_result_py = None
        self.current_plot_pixmap = None
        self.result_text_preview.setPlainText(message)
        for button in (self.download_txt_button, self.download_png_button,
                       self.download_py_button, self.open_results_folder_button):
            button.setEnabled(False)
        self._set_plot_click_enabled(False)

    def save_emotion_result(self, kind):
        if kind in self.emotion_results and not self.analysis_in_progress:
            path = Path(self.emotion_results[kind])
            suffix = path.suffix
            self._save_result_copy(path, f'{suffix[1:].upper()} (*{suffix})')

    def refresh_emotion_preview(self):
        for preview, key in ((self.emotion_preview, 'png'), (self.density_preview, 'density_png')):
            if self.emotion_results:
                pixmap = self._load_png_pixmap(Path(self.emotion_results[key]))
                preview.setPixmap(pixmap.scaled(preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))
            else:
                preview.clear()
                preview.setText("Выберите текст и выполните анализ")

    @Slot()
    def shutdown(self) -> None:
        if self.thread is None:
            return
        try:
            if self.thread.isRunning():
                self.thread.quit()
                if not self.thread.wait(5000):
                    self.thread.wait()
        except RuntimeError:
            # Thread object can already be deleted during teardown.
            pass



class PackagedSmokeReceiver(QObject):
    def __init__(self, controller, app, thread, report):
        super().__init__(controller.window)
        self.controller, self.app, self.thread, self.report = controller, app, thread, report

    @Slot(dict)
    def complete(self, artifacts):
        import json
        try:
            self.controller.on_analysis_finished(artifacts)
            if artifacts.get("emotions"):
                self.controller.plot_tabs.setCurrentIndex(1)
            self.app.processEvents()
            self.controller.window.grab().save(str(self.report.with_suffix(".png")))
            if artifacts.get("emotions"):
                if 'Преобладающ' not in self.controller.emotion_report.toPlainText():
                    raise RuntimeError("Packaged marker report was not rendered")
                self.controller.plot_tabs.setCurrentIndex(3)
                self.app.processEvents()
                self.controller.window.grab().save(str(self.report.with_name(self.report.stem + "_markers.png")))
            from scripts import emotions
            artifacts['dictionary'] = dict(sha256=emotions.SOURCE_SHA256,
                                           entries=emotions.SOURCE_ROWS, categories=list(emotions.LABELS),
                                           source_exists=(emotions.DATA_DIR / 'NRC2022.SOURCE.md').is_file(),
                                           checksum_exists=(emotions.DATA_DIR / 'NRC2022.sha256').is_file(),
                                           translation=emotions.source_metadata()['translation'],
                                           readme_exists=(emotions.DATA_DIR / 'nrc2022' / 'README.txt').is_file(),
                                           profile=emotions.profile_metadata(artifacts['emotion_profile']),
                                           offline=os.environ.get('SENTILEX_SMOKE_OFFLINE') == '1')
            self.controller.plot_tabs.setCurrentIndex(2)
            self.app.processEvents()
            self.controller.window.grab().save(str(self.report.with_name(self.report.stem + "_density.png")))
            self.report.write_text(json.dumps(artifacts, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self.failed(str(exc))
        finally:
            self.thread.quit()

    @Slot(str)
    def failed(self, message):
        import json
        self.report.write_text(json.dumps({"error": message}, ensure_ascii=False), encoding="utf-8")
        self.thread.quit()


def load_window() -> QMainWindow:
    ui_path = resource_path("desktop", "ui", "main.ui")
    loader = QUiLoader()
    ui_file = QFile(str(ui_path))
    if not ui_file.open(QFile.ReadOnly):
        raise RuntimeError(f"Cannot open UI file: {ui_path}")
    widget = loader.load(ui_file)
    ui_file.close()
    if widget is None:
        raise RuntimeError(f"Cannot load UI file: {ui_path}; loader error: {loader.errorString()}")

    if isinstance(widget, QMainWindow):
        return widget

    window = QMainWindow()
    window.setCentralWidget(widget)
    return window


def fit_window_to_screen(window: QMainWindow, app: QApplication) -> None:
    screen = app.primaryScreen()
    if screen is None:
        window.resize(1360, 860)
        return

    available = screen.availableGeometry()
    target_width = min(1600, max(960, int(available.width() * 0.94)))
    target_height = min(980, max(680, int(available.height() * 0.90)))
    target_width = min(target_width, available.width())
    target_height = min(target_height, available.height())

    x_pos = available.x() + max(0, (available.width() - target_width) // 2)
    y_pos = available.y() + max(0, (available.height() - target_height) // 2)
    window.setMinimumSize(820, 620)
    window.setGeometry(x_pos, y_pos, target_width, target_height)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = load_window()
    qss_path = resource_path("desktop", "style.qss")
    window.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    controller = MainWindowController(window)
    app.aboutToQuit.connect(controller.shutdown)
    fit_window_to_screen(window, app)
    window.show()

    window._controller = controller  # type: ignore[attr-defined]
    # Opt-in packaged smoke exercises frozen resources and the real worker/UI.
    if os.environ.get("SENTILEX_SMOKE_INPUT"):
        import json
        report = Path(os.environ["SENTILEX_SMOKE_REPORT"])
        smoke_profile = os.environ.get("SENTILEX_SMOKE_PROFILE", "literary")
        controller.emotion_profile.setCurrentIndex(controller.emotion_profile.findData(smoke_profile))
        worker = AnalysisWorker(Path(os.environ["SENTILEX_SMOKE_INPUT"]), 50,
                                emotion_profile=smoke_profile)
        thread = QThread(window)
        worker.moveToThread(thread)
        receiver = PackagedSmokeReceiver(controller, app, thread, report)
        worker.finished.connect(receiver.complete)
        worker.failed.connect(receiver.failed)
        thread.started.connect(worker.run)
        thread.finished.connect(app.quit)
        window._smoke_worker, window._smoke_thread = worker, thread
        thread.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
