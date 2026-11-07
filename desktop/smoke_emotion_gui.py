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
    worker = AnalysisWorker(source, 1, str(Path('NRC-Suite-of-Sentiment-Emotion-Lexicons.zip').resolve()))
    errors, results = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(results.append)
    worker.run()
    assert not errors, errors
    controller.on_analysis_finished(results[0])
    assert len(controller.emotion_results) == 3
    controller.plot_tabs.setCurrentIndex(1)
    app.processEvents()
    assert not controller.emotion_preview.pixmap().isNull()
    controller.content_scroll_area.verticalScrollBar().setValue(controller.content_scroll_area.verticalScrollBar().maximum())
    app.processEvents()
    window.grab().save('/tmp/sentilex-emotion-gui.png')
    # A subsequent sentiment-only result must remove prior emotion results.
    without = dict(results[0])
    without.pop('emotions')
    controller.on_analysis_finished(without)
    assert not controller.emotion_results
    assert controller.emotion_preview.pixmap() is None or controller.emotion_preview.pixmap().isNull()
window.close()
print('[smoke-emotion-gui] OK')
