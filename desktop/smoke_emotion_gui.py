"""Real threaded desktop flow: bundled dictionary, defaults, save, refresh and errors."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QSettings, QEventLoop, QTimer
from PySide6.QtWidgets import QApplication
from main import MainWindowController, load_window, resource_path
from scripts import emotions

app = QApplication([])
settings_folder = tempfile.TemporaryDirectory()
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, settings_folder.name)
settings = QSettings('Sentilex', 'Sentilex')
settings.setValue('nrc_path', 'Z:/missing/obsolete-NRC.xlsx')
settings.sync()
window = load_window()
window.setStyleSheet(resource_path('desktop', 'style.qss').read_text(encoding='utf-8'))
controller = MainWindowController(window)
window.resize(1600, 1100)
window.show()


def choose(source):
    with patch('main.QFileDialog.getOpenFileName', return_value=(str(source), '')):
        controller.choose_file()
    assert controller.chunk_number.value() == 1
    assert not controller.emotion_results
    assert all(not b.isEnabled() for b in controller.emotion_save_buttons.values())


def analyze(size):
    controller.chunk_number.setValue(size)
    print("Analyzing", controller.selected_file, size, flush=True)
    controller.start_analysis()
    assert not controller.emotion_results
    assert all(not b.isEnabled() for b in controller.emotion_save_buttons.values())
    loop = QEventLoop()
    timeout = QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    controller.thread.finished.connect(loop.quit)
    timeout.start(90000)
    loop.exec()
    timeout.stop()
    assert not controller.analysis_in_progress, 'Worker timed out'
    app.processEvents()


try:
    assert 'RusEmoLex' in window.windowTitle()
    assert not hasattr(controller, 'emotion_path') and not hasattr(controller, 'nrc_button')
    assert not controller.chunk_number.isEnabled() and not controller.chunk_slider.isEnabled()
    window.grab().save('/tmp/sentilex-rusemolex-start.png')
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder) / 'rusemolex_gui.txt'
        source.write_text('Я испугался. Радость и счастье. Стол и окно.', encoding='utf-8')
        choose(source)
        analyze(1)
        assert len(controller.emotion_results) == 6
        assert 'Найдено: «испугался» → Страх' in controller.emotion_report.toPlainText()
        assert 'В этом предложении не найдены слова из словаря' in controller.emotion_report.toPlainText()
        assert all(b.isEnabled() for b in controller.emotion_save_buttons.values())
        for kind, button in controller.emotion_save_buttons.items():
            original = Path(controller.emotion_results[kind])
            destination = Path(folder) / original.name
            with patch('main.QFileDialog.getSaveFileName', return_value=(str(destination), '')):
                button.click()
            assert destination.read_bytes() == original.read_bytes()
        for index, name in ((1, 'clusters'), (2, 'density'), (3, 'markers')):
            controller.plot_tabs.setCurrentIndex(index)
            app.processEvents()
            controller.content_scroll_area.verticalScrollBar().setValue(controller.content_scroll_area.verticalScrollBar().maximum())
            app.processEvents()
            window.grab().save(f'/tmp/sentilex-rusemolex-{name}-gui.png')
        assert not controller.emotion_preview.pixmap().isNull()
        assert not controller.density_preview.pixmap().isNull()
        first = {k: Path(p).stat().st_mtime_ns for k, p in controller.emotion_results.items()}
        analyze(2)
        assert len(controller.emotion_results) == 6
        assert all(Path(p).stat().st_mtime_ns > first[k] for k, p in controller.emotion_results.items())
        assert 'Фрагмент 2' in controller.emotion_report.toPlainText()
        assert 'Фрагмент 3' not in controller.emotion_report.toPlainText()
        # Failure after creating one partial artifact must remove all emotion files.
        stale_paths = list(controller.emotion_results.values())
        def fail(*args):
            Path(stale_paths[0]).write_text('partial')
            raise ValueError('synthetic failure')
        with patch('scripts.emotions.write_results', side_effect=fail), patch('main.QMessageBox.critical'):
            analyze(3)
        assert not controller.emotion_results and all(not Path(p).exists() for p in stale_paths)
        assert all(not b.isEnabled() for b in controller.emotion_save_buttons.values())
        assert controller.current_result_txt is None and controller.current_result_png is None
        assert controller.density_preview.pixmap() is None or controller.density_preview.pixmap().isNull()
        assert 'Преобладающ' not in controller.emotion_report.toPlainText()
        # Actual supplied chapter: reselect resets to 1; sizes 1/2/3 produce every sentence.
        chapter = Path('Chapter_7_Karenina.txt')
        if chapter.exists():
            choose(chapter.resolve())
            assert controller.selected_sentence_count == 94
            for size in (1, 2, 3):
                analyze(size)
                assert controller.emotion_results
                import csv
                with open(controller.emotion_results['sentences_csv'], encoding='utf-8-sig') as f:
                    assert len(list(csv.reader(f, delimiter=';'))) == 95
            for index, name in ((1, 'chapter-clusters'), (2, 'chapter-density'), (3, 'chapter-markers')):
                controller.plot_tabs.setCurrentIndex(index)
                app.processEvents()
                window.grab().save(f'/tmp/sentilex-rusemolex-{name}-gui.png')
        dense = Path(folder) / 'rusemolex_dense.txt'
        dense.write_text(' '.join(['Злость грусть радость страх удивление.'] * 240), encoding='utf-8')
        choose(dense)
        analyze(1)
        assert controller.emotion_results
        controller.plot_tabs.setCurrentIndex(2)
        app.processEvents()
        window.grab().save('/tmp/sentilex-rusemolex-dense-gui.png')
finally:
    print("Closing GUI", flush=True)
    controller.shutdown()
    window.close()
    settings_folder.cleanup()
print('[smoke-emotion-gui] OK: threaded analysis, six saved files, rerun/error clearing, chapter 1/2/3 and dense text')
