"""Dense chapter labels, complete legend and runtime/export parity."""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app
from scripts.export_script import build_export_script
import matplotlib.pyplot as plt

text = build_export_script(default_chunk_size=10, lexicon_path='scripts/RuSentilex-2017.txt', chapter_label_step=20)
namespace = {'__name__': 'plot_options_test'}
exec(compile(text, 'standalone', 'exec'), namespace)
assert namespace['CHAPTER_LABEL_STEP'] == 20
values = [((i % 39) - 19) / 100 for i in range(240)]
metrics = []
with tempfile.TemporaryDirectory() as folder:
    for index, function in enumerate((app.functions.plot_sentiment_curve, namespace['plot_sentiment_curve'])):
        with patch.object(plt, 'close'):
            result = function(values, ['Текст.'] * 240, list(range(1, 241)), [f'Глава {i}' for i in range(1, 241)], 10, str(Path(folder) / f'{index}.png'), **({"chapter_label_step": 20} if index == 0 else {}))
            figure = plt.gcf()
        assert len(figure.axes[0].get_xticks()) == 12
        labels = [item.get_text() for item in figure.axes[0].get_legend().get_texts()]
        assert len(labels) == 12, labels
        assert any('Максимум исходной' in label for label in labels)
        assert any('границы коридора' in label for label in labels)
        figure.savefig('/tmp/sentilex-chapter-stride.png', bbox_inches='tight')
        plt.close(figure)
        metrics.append(result)
assert metrics[0] == metrics[1]
print('[smoke-plot-options] OK: dense labels thinned, legend complete, export parity')
