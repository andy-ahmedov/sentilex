"""Offscreen NRC worker, result tabs and reanalysis refresh check."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
from pathlib import Path
from PySide6.QtWidgets import QApplication
from main import AnalysisWorker, MainWindowController, load_window, resource_path

app = QApplication([])
window = load_window()
window.setStyleSheet(resource_path('desktop', 'style.qss').read_text())
controller = MainWindowController(window)
window.resize(1500, 1000)
window.show()
with tempfile.TemporaryDirectory() as folder:
    source = Path(folder) / 'emotion_gui.txt'
    source.write_text('Страх и ужас. Радость и счастье. Стол и окно.', encoding='utf-8')
    controller.selected_file = source
    controller._configure_chunk_slider(3)
    controller.chunk_number.setValue(1)
    controller.file_path_edit.setText(str(source))
    controller.upload_label.setText(source.name)
    worker = AnalysisWorker(source, 1, str(Path('NRC-Suite-of-Sentiment-Emotion-Lexicons.zip').resolve()))
    errors, results = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(results.append)
    worker.run()
    assert not errors, errors
    controller.on_analysis_finished(results[0])
    assert len(controller.emotion_results) == 4
    assert 'Лидеры предложения' in controller.emotion_report.toPlainText()
    assert 'Маркер' in controller.emotion_report.toPlainText() or '«' in controller.emotion_report.toPlainText()
    assert controller.save_cluster_html_button.isEnabled()
    from unittest.mock import patch
    with patch.object(controller, '_save_result_copy') as save_copy:
        controller.save_cluster_html_button.click()
        assert save_copy.call_args.args[0] == Path(results[0]['emotions']['html'])
    controller.plot_tabs.setCurrentIndex(1)
    app.processEvents()
    assert not controller.emotion_preview.pixmap().isNull()
    controller.content_scroll_area.verticalScrollBar().setValue(controller.content_scroll_area.verticalScrollBar().maximum())
    app.processEvents()
    window.grab().save('/tmp/sentilex-emotion-gui.png')
    controller.plot_tabs.setCurrentIndex(2)
    app.processEvents()
    assert controller.emotion_report.isVisible()
    window.grab().save('/tmp/sentilex-cluster-report-gui.png')
    # A subsequent sentiment-only result must remove prior emotion results.
    without = dict(results[0])
    without.pop('emotions')
    controller.on_analysis_finished(without)
    assert not controller.emotion_results
    assert controller.emotion_preview.pixmap() is None or controller.emotion_preview.pixmap().isNull()
    assert 'Лидеры предложения' not in controller.emotion_report.toPlainText()
    assert not controller.save_cluster_html_button.isEnabled()
    # A failed rerun must clear previous details too.
    controller.on_analysis_finished(results[0])
    with patch('main.QMessageBox.critical'):
        controller.on_analysis_failed('synthetic failure')
    assert not controller.emotion_results
    assert 'Лидеры предложения' not in controller.emotion_report.toPlainText()
    assert not controller.save_cluster_html_button.isEnabled()
window.close()
print('[smoke-emotion-gui] OK')
